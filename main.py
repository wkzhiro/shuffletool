import streamlit as st
import pandas as pd
import numpy as np
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
from dotenv import load_dotenv
from collections import Counter

warnings.filterwarnings('ignore')

# 環境変数の読み込み
load_dotenv('.env.local')

def can_convert_to_float(x):
    """xが浮動小数点数に変換可能ならTrue, それ以外False"""
    try:
        float(x)
        return True
    except:
        return False

def group_selection(unique_keys, label):
    """
    1つのカテゴリについて、ユニークキーをグループ分けする処理。
    ここでは数値かどうか判定しつつ、すべて文字列としてソートしています。
    """
    # まずNaNは除外
    unique_keys = [x for x in unique_keys if pd.notna(x)]

    # すべて文字列に変換し、前後の空白を削除（strip）
    unique_keys_str = [str(x).strip() for x in unique_keys]

    # 数値判定 → 今回は例として「数値ソート or 文字列ソート」残しますが
    # 全部文字列ソートにする場合はコメントアウトしてください
    if all(can_convert_to_float(x) for x in unique_keys_str):
        sorted_keys = sorted(unique_keys_str, key=lambda x: float(x))
    else:
        sorted_keys = sorted(unique_keys_str)

    groups = []
    remaining_keys = sorted_keys.copy()

    st.write(f"### {label} グループ分け")
    # st.write("#### ユニークキー (ソート後) =", remaining_keys)

    for i in range(5):
        group = st.multiselect(
            f"Group {i + 1} ({label})",
            options=remaining_keys,
            default=[]
        )
        groups.append(group)
        # 選択されたキーを remaining_keys から除外
        remaining_keys = [k for k in remaining_keys if k not in group]
        if not remaining_keys:
            break

    # # デバッグ: どのグループにどんなキーが入ったか
    # st.write(f"=== DEBUG: {label} groups ===", groups)
    return groups

def assign_groups(df_col, groups):
    """
    例: groups=[["A","B"],["C"],["D","E"]]
     => "A","B" は catX_group="Group 1"
        "C" は catX_group="Group 2"
        "D","E" は catX_group="Group 3"
    """
    # df_col も全て文字列＋stripしておく
    df_col_str = df_col.astype(str).str.strip()

    group_map = {}
    for idx, group_list in enumerate(groups):
        for key in group_list:
            # key 自体も strip() しておくと安全
            k = key.strip()
            group_map[k] = f"Group {idx + 1}"

    # st.write("=== DEBUG: group_map ===")
    # st.write(group_map)

    # マッピング
    mapped_series = df_col_str.map(group_map)

    # # デバッグ: 実際にマッピングされた結果がどうなったか表示
    # st.write("=== DEBUG: assign_groups result (sample) ===")
    # st.write(pd.DataFrame({
    #     "original": df_col.head(20),      # 元のExcel列 (そのまま)
    #     "original_stripped": df_col_str.head(20),  # strip後
    #     "mapped": mapped_series.head(20)
    # }))

    return mapped_series

def get_embeddings(texts, client, deployment_name):
    """
    Azure OpenAIを使用してテキストのembeddingを取得
    """
    if client is None:
        st.error("Azure OpenAIクライアントが初期化されていません")
        return None
        
    try:
        embeddings = []
        for text in texts:
            response = client.embeddings.create(
                input=str(text),
                model=deployment_name
            )
            embeddings.append(response.data[0].embedding)
        return np.array(embeddings)
    except Exception as e:
        st.error(f"Embedding取得エラー: {e}")
        return None

def analyze_column_type(series):
    """
    列のデータタイプを自動判定（フリーテキスト vs タグ系）
    """
    # NaNを除外
    valid_data = series.dropna()
    if len(valid_data) == 0:
        return "tag", "空のデータ"
    
    # 数値列の場合は明らかにタグ系
    if pd.api.types.is_numeric_dtype(series):
        return "tag", "数値データ"
    
    # 文字列データの分析
    texts = valid_data.astype(str)
    
    # 1. 文字列長分析
    avg_length = texts.str.len().mean()
    
    # 2. ユニーク値の割合
    unique_ratio = len(texts.unique()) / len(texts)
    
    # 3. 長いテキストの割合（30文字以上）
    long_text_ratio = (texts.str.len() >= 30).mean()
    
    # 4. 句読点を含む割合
    punctuation_ratio = texts.str.contains(r'[。、！？.,!?]', regex=True).mean()
    
    # 5. 空白を含む割合（複数単語）
    space_ratio = texts.str.contains(r'\s+', regex=True).mean()
    
    # 判定ロジック
    free_text_score = 0
    reasons = []
    
    if avg_length >= 20:
        free_text_score += 2
        reasons.append(f"平均文字数: {avg_length:.1f}")
    
    if unique_ratio >= 0.7:
        free_text_score += 2
        reasons.append(f"ユニーク値割合: {unique_ratio:.1%}")
    
    if long_text_ratio >= 0.3:
        free_text_score += 2
        reasons.append(f"長文割合: {long_text_ratio:.1%}")
    
    if punctuation_ratio >= 0.3:
        free_text_score += 1
        reasons.append(f"句読点含有率: {punctuation_ratio:.1%}")
    
    if space_ratio >= 0.5:
        free_text_score += 1
        reasons.append(f"空白含有率: {space_ratio:.1%}")
    
    # 判定結果
    if free_text_score >= 4:
        return "free_text", f"フリーテキスト判定 (スコア:{free_text_score}) - " + ", ".join(reasons)
    else:
        return "tag", f"タグ系判定 (スコア:{free_text_score}) - " + ", ".join(reasons)

def auto_detect_text_columns(df):
    """
    データフレーム内のフリーテキスト列を自動検出
    """
    text_columns = []
    analysis_results = {}
    
    for i, col in enumerate(df.columns):
        if i == 0:  # 最初の列（通常はID）はスキップ
            continue
            
        col_type, reason = analyze_column_type(df[col])
        analysis_results[f"列{i+1}: {col}"] = {
            "type": col_type,
            "reason": reason,
            "index": i
        }
        
        if col_type == "free_text":
            text_columns.append(i)
    
    return text_columns, analysis_results

def calculate_text_similarity_groups(text_series, client, deployment_name):
    """
    Azure OpenAIのembeddingを使用してフリーテキストの類似度を計算し、仮想グループを作成
    """
    # NaNを空文字列に置換
    texts = text_series.fillna("").astype(str)
    
    # 空文字列や短すぎるテキストの場合のフォールバック
    valid_texts = [t for t in texts if len(t.strip()) > 0]
    if len(valid_texts) < 2:
        return pd.Series(["TextGroup 1"] * len(texts), index=text_series.index)
    
    # Azure OpenAIでembeddingを取得
    st.info("テキストの埋め込みベクトルを計算中...")
    embeddings = get_embeddings(texts, client, deployment_name)
    
    if embeddings is None:
        st.warning("embedding計算に失敗しました。全員を同じグループにします。")
        return pd.Series(["TextGroup 1"] * len(texts), index=text_series.index)
    
    try:
        # コサイン類似度行列を計算
        similarity_matrix = cosine_similarity(embeddings)
        
        # 簡易クラスタリング（閾値ベース）
        similarity_threshold = 0.7  # Azure OpenAIのembeddingは高品質なので閾値を高めに設定
        text_groups = pd.Series(["TextGroup Other"] * len(texts), index=text_series.index)
        group_counter = 1
        assigned = set()
        
        for i in range(len(texts)):
            if i in assigned:
                continue
                
            # 現在のテキストと類似したテキストを探す
            similar_indices = [i]
            for j in range(i + 1, len(texts)):
                if j not in assigned and similarity_matrix[i][j] > similarity_threshold:
                    similar_indices.append(j)
            
            # グループに割り当て
            if len(similar_indices) >= 2:  # 2人以上でグループ形成
                group_name = f"TextGroup {group_counter}"
                for idx in similar_indices:
                    text_groups.iloc[idx] = group_name
                    assigned.add(idx)
                group_counter += 1
            else:
                assigned.add(i)
        
        return text_groups
        
    except Exception as e:
        st.warning(f"テキスト類似度計算エラー: {e}")
        return pd.Series(["TextGroup 1"] * len(texts), index=text_series.index)

def create_advanced_similarity_matrix(df, text_col1, text_col2=None, client=None, deployment_name=None):
    """
    【修正版】高度グルーピング用の類似度行列を作成
    ★ Embeddingの結果を表示する機能を追加
    """
    st.info("類似度行列を作成中...")
    
    # テキスト準備
    texts = []
    for idx in df.index:
        if text_col2 is None:
            # 単一列での類似度
            text = str(df.iloc[idx, text_col1]) if pd.notna(df.iloc[idx, text_col1]) else ""
            texts.append(text)
        else:
            # 2列結合での類似度
            text1 = str(df.iloc[idx, text_col1]) if pd.notna(df.iloc[idx, text_col1]) else ""
            text2 = str(df.iloc[idx, text_col2]) if pd.notna(df.iloc[idx, text_col2]) else ""
            combined_text = f"{text1} [SEP] {text2}"
            texts.append(combined_text)
    
    # Azure OpenAIを試す
    if client and deployment_name:
        try:
            st.write("Azure OpenAI APIを呼び出してEmbeddingを取得します...")
            embeddings = get_embeddings_batch(texts, client, deployment_name)
            
            if embeddings is not None:
                # --- ここからが追加部分 ---
                st.success(f"✅ Embeddingの取得に成功しました。")
                st.info(f"取得したベクトル数: {embeddings.shape[0]}, 各ベクトルの次元数: {embeddings.shape[1]}")

                # 類似度行列を計算
                similarity_matrix = cosine_similarity(embeddings)

                # 詳細を見たい人向けに、ベクトルと類似度行列を展開表示
                with st.expander("クリックしてEmbeddingと類似度行列の詳細を確認"):
                    st.write("▼ 取得したEmbeddingベクトル（最初の5件）:")
                    st.dataframe(pd.DataFrame(embeddings).head())
                    st.write("▼ 計算された類似度行列（最初の5x5部分）:")
                    st.dataframe(pd.DataFrame(similarity_matrix).head())
                # --- 追加部分ここまで ---
                
                mode = "単一列" if text_col2 is None else "2列結合"
                st.success(f"Azure OpenAIによる類似度行列の作成が完了しました。（{mode}）")
                return similarity_matrix
        except Exception as e:
            st.warning(f"Azure OpenAI処理エラー: {e}. TF-IDFにフォールバックします。")
    
    # TF-IDFフォールバック
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        st.write("TF-IDFによる類似度計算にフォールバックします...")
        
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        mode = "単一列" if text_col2 is None else "2列結合"
        st.success(f"TF-IDFによる類似度行列を作成しました（{mode}）")
        return similarity_matrix
    except Exception as e:
        st.error(f"類似度行列の作成に失敗しました: {e}")
        return None

def get_embeddings_batch(texts, client, deployment_name, batch_size=10):
    """バッチでembeddingを取得"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model=deployment_name,
                input=batch_texts
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            st.error(f"バッチ {i//batch_size + 1} の処理でエラー: {e}")
            return None
    
    return np.array(all_embeddings)

def calculate_advanced_group_score(df, group_indices, similarity_matrix, priority_options, score_map, category_strategies, w_label=1.0, w_sim=1.0):
    """
    【修正版】高度グルーピングのスコア計算
    カテゴリマッチングで部分点を考慮するロジックを追加
    """
    if len(group_indices) < 2:
        return 0

    # 1. カテゴリマッチングスコア計算
    category_score = 0
    for priority in priority_options:
        if priority not in score_map:
            continue
            
        base_points = score_map[priority]
        
        cat_num = int(priority.replace('cat', '').replace('_group', ''))
        strategy = category_strategies.get(cat_num, 'diversity')
        
        col_index = df.columns.get_loc(priority)
        values = [df.iloc[idx, col_index] for idx in group_indices]
        
        # NaNなどを除外した有効な値のリストを作成
        valid_values = [v for v in values if pd.notna(v)]
        if len(valid_values) < 2:
            continue
            
        unique_values = len(set(valid_values))
        total_values = len(valid_values)
        
        # --- ここからが修正・確認部分です ---
        if strategy == 'diversity':
            # 多様性重視：全員異なるなら満点、ユニークな値が半数以上なら半分の点数
            if unique_values == total_values:
                category_score += base_points  # 満点
            elif unique_values / total_values > 0.75:
                category_score += base_points / 2  # 半分の点数を加算

        elif strategy == 'homogeneity':
            # 同質性重視：全員同じ値なら満点、単一の値が半数以上なら半分の点数
            if unique_values == 1:
                category_score += base_points  # 満点
            else:
                # 最も多いカテゴリのメンバー数を取得
                counts = Counter(valid_values)
                max_count = counts.most_common(1)[0][1]

                # 最も多いカテゴリが全体の半数を超えているかチェック
                if max_count / total_values > 0.75:
                    category_score += base_points / 2  # 半分の点数を加算
    
    # 2. テキスト類似度スコア計算
    similarity_sum = 0
    for i, idx1 in enumerate(group_indices):
        for idx2 in group_indices[i+1:]:
            similarity_sum += similarity_matrix[idx1][idx2]
    
    # 3. 総合スコア
    total_score = (w_label * category_score) + (w_sim * similarity_sum)
    return total_score

def find_advanced_optimal_groups(df, similarity_matrix, priority_options, score_map, category_strategies, target_group_size=4, w_sim=1.0):
    """
    【修正版】反復最適化による高度グルーピング
    指定された人数でグループを優先的に作成し、最後に端数処理を行う。
    """
    st.info(f"希望グループサイズ {target_group_size}人での最適化を開始...")

    available_indices = list(df.index)
    final_groups = []
    group_id_counter = 1
    
    # 組み合わせの探索回数の上限を設定（計算時間を現実的にするため）
    max_combinations = 2000

    # --- 修正ポイント1: まず `target_group_size` でグループを作り続ける ---
    while len(available_indices) >= target_group_size:
        st.write(f"--- {target_group_size}人グループの探索 (残り {len(available_indices)}人) ---")
        best_group = None
        best_score = -float('inf')

        # 全員の組み合わせを試すと計算量が膨大になるため、候補者をサンプリングして探索する
        from itertools import combinations
        import random
        
        sample_size = min(len(available_indices), 40) # 探索対象を最大40人に絞る
        if sample_size < target_group_size:
            search_pool = available_indices
        else:
            search_pool = random.sample(available_indices, k=sample_size)

        combinations_count = 0
        for group_indices in combinations(search_pool, target_group_size):
            combinations_count += 1
            if combinations_count > max_combinations:
                st.warning(f"組み合わせが多すぎるため、{max_combinations}回で探索を打ち切りました。")
                break
            
            score = calculate_advanced_group_score(
                df, list(group_indices), similarity_matrix,
                priority_options, score_map, category_strategies, 1.0, w_sim
            )

            if score > best_score:
                best_score = score
                best_group = list(group_indices)

        # 最適なグループが見つからなかった場合は、強制的にグループを作成する
        if best_group is None:
            st.warning("最適なグループが見つかりませんでした。先頭から強制的にグループを作成します。")
            best_group = available_indices[:target_group_size]
            best_score = calculate_advanced_group_score(
                df, best_group, similarity_matrix,
                priority_options, score_map, category_strategies, 1.0, w_sim
            )

        # 見つかった最適なグループを確定する
        final_groups.append({
            'members': best_group,
            'score': best_score,
            'size': len(best_group)
        })
        for idx in best_group:
            available_indices.remove(idx)
        st.write(f"✅ グループ {group_id_counter} ({len(best_group)}人) を確定 (スコア: {best_score:.2f})")
        group_id_counter += 1

    # --- 修正ポイント2: 端数処理 ---
    st.info(f"端数処理中... (残り {len(available_indices)}人)")
    
    # 最小グループ人数を定義（これより少ない場合は既存グループに吸収）
    # 例: targetが4なら3人、targetが3なら2人
    min_final_group_size = max(2, target_group_size - 1)

    if len(available_indices) >= min_final_group_size:
        # 残ったメンバーで1つのグループを作成
        rem_group = available_indices.copy()
        score = calculate_advanced_group_score(
            df, rem_group, similarity_matrix,
            priority_options, score_map, category_strategies, 1.0, w_sim
        )
        final_groups.append({
            'members': rem_group,
            'score': score,
            'size': len(rem_group)
        })
        st.write(f"✅ グループ {group_id_counter} ({len(rem_group)}人, 端数グループ) を確定 (スコア: {score:.2f})")
        available_indices.clear()

    # --- 修正ポイント3: それでも残った少数のメンバーを既存グループに割り当てる ---
    if available_indices:
        st.info(f"最終的な残りメンバー {len(available_indices)}人を既存グループに割り当てます。")
        assign_remaining_members_advanced(df, available_indices, final_groups, similarity_matrix, priority_options, score_map, category_strategies, w_sim)

    st.success(f"最適化完了: {len(final_groups)} グループが作成されました")
    return final_groups

def assign_remaining_members_advanced(df, remaining_indices, groups, similarity_matrix, priority_options, score_map, category_strategies, w_sim):
    """
    【修正版】残存メンバーを既存グループに割り当て
    ★ 引数の不一致を修正
    """
    st.info(f"残り {len(remaining_indices)} 人を既存グループに割り当て中...")
    
    for member_idx in remaining_indices:
        best_group_idx = -1
        best_score_increase = -float('inf')
        
        for group_idx, group in enumerate(groups):
            # メンバーを追加した場合のスコア増分を計算
            original_score = group.get('score', 0) # scoreキーがない場合も考慮
            extended_group = group['members'] + [member_idx]
            
            # calculate_advanced_group_scoreの引数を修正済みのものに合わせる
            new_score = calculate_advanced_group_score(
                df, extended_group, similarity_matrix, 
                priority_options, score_map, category_strategies, 1.0, w_sim
            )
            score_increase = new_score - original_score
            
            if score_increase > best_score_increase:
                best_score_increase = score_increase
                best_group_idx = group_idx
        
        if best_group_idx >= 0:
            groups[best_group_idx]['members'].append(member_idx)
            # グループのスコアとサイズも更新する
            groups[best_group_idx]['score'] = calculate_advanced_group_score(
                df, groups[best_group_idx]['members'], similarity_matrix,
                priority_options, score_map, category_strategies, 1.0, w_sim
            )
            groups[best_group_idx]['size'] += 1
            st.write(f"メンバー追加: グループ {best_group_idx + 1} に割り当て (スコア増分: {best_score_increase:.2f})")

def create_advanced_results_dataframe(df, groups):
    """高度グルーピング結果をDataFrameとして生成"""
    if not groups:
        return None
    
    result_df = df.copy()
    result_df['advanced_group_id'] = 0
    result_df['advanced_group_score'] = 0.0
    
    for group_idx, group in enumerate(groups):
        for member_idx in group['members']:
            result_df.loc[member_idx, 'advanced_group_id'] = group_idx + 1
            result_df.loc[member_idx, 'advanced_group_score'] = group['score']
    
    return result_df

def display_advanced_group_analysis(df, groups, priority_options, score_map, category_strategies):
    """高度グルーピング結果の分析表示"""
    if not groups:
        st.warning("グループが作成されていません")
        return
    
    st.subheader("📊 高度グルーピング分析結果")
    
    # 戦略情報の表示
    st.write("**適用された戦略:**")
    strategy_info = []
    for cat_num, strategy in category_strategies.items():
        strategy_name = "多様性重視" if strategy == "diversity" else "同質性重視"
        strategy_info.append(f"カテゴリ{cat_num}: {strategy_name}")
    st.write(" | ".join(strategy_info))
    
    # 基本統計
    total_members = sum(group['size'] for group in groups)
    avg_group_size = total_members / len(groups)
    total_score = sum(group['score'] for group in groups)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("総グループ数", len(groups))
    with col2:
        st.metric("総メンバー数", total_members)
    with col3:
        st.metric("平均グループサイズ", f"{avg_group_size:.1f}")
    with col4:
        st.metric("総合スコア", f"{total_score:.1f}")
    
    # 各グループの詳細
    for idx, group in enumerate(groups):
        with st.expander(f"グループ {idx + 1} (スコア: {group['score']:.2f}, {group['size']}人)"):
            group_df = df.iloc[group['members']]
            st.dataframe(group_df)
            
            # カテゴリ別分析
            st.write("**カテゴリ別分析:**")
            for priority in priority_options:
                if priority in df.columns:
                    col_values = group_df[priority].value_counts()
                    cat_num = int(priority.replace('cat', '').replace('_group', ''))
                    strategy = category_strategies.get(cat_num, 'diversity')
                    strategy_name = "多様性重視" if strategy == "diversity" else "同質性重視"
                    
                    st.write(f"- {priority} ({strategy_name}): {dict(col_values)}")

def main():
    st.title("カテゴリ優先度付きシャッフル + フリーテキスト類似度対応")
    
    # Azure OpenAI設定（環境変数から取得）
    ENDPOINT = os.getenv("ENDPOINT")
    API_KEY = os.getenv("API_KEY")
    
    # ENDPOINTからbase URLを抽出
    if ENDPOINT:
        # https://openai-shuffle.openai.azure.com/openai/deployments/text-embedding-3-large/embeddings?api-version=2023-05-15
        # から https://openai-shuffle.openai.azure.com/ を抽出
        import re
        match = re.match(r'(https://[^/]+)', ENDPOINT)
        AZURE_ENDPOINT = match.group(1) + "/" if match else None
        
        # API versionとdeployment nameを抽出
        if 'api-version=' in ENDPOINT:
            API_VERSION = ENDPOINT.split('api-version=')[1]
        else:
            API_VERSION = "2023-05-15"
            
        if '/deployments/' in ENDPOINT:
            DEPLOYMENT_NAME = ENDPOINT.split('/deployments/')[1].split('/')[0]
        else:
            DEPLOYMENT_NAME = "text-embedding-3-large"
    else:
        AZURE_ENDPOINT = None
        API_VERSION = "2023-05-15"
        DEPLOYMENT_NAME = "text-embedding-3-large"
    
    # 環境変数の確認
    if not ENDPOINT or not API_KEY:
        st.error("⚠️ 環境変数が設定されていません。.env.localファイルを確認してください。")
        st.write("必要な環境変数:")
        st.write("- ENDPOINT")
        st.write("- API_KEY")
        client = None
    else:
        # Azure OpenAIクライアント初期化
        try:
            client = AzureOpenAI(
                azure_endpoint=AZURE_ENDPOINT,
                api_key=API_KEY,
                api_version=API_VERSION
            )
            st.success(f"✅ Azure OpenAIクライアントが正常に初期化されました")
            st.info(f"使用中のデプロイメント: {DEPLOYMENT_NAME}")
        except Exception as e:
            st.error(f"❌ Azure OpenAIクライアントの初期化に失敗しました: {e}")
            client = None

    # 1) エクセルファイルのアップロード
    uploaded_file = st.file_uploader("エクセルファイルをアップロードしてください", type=["xlsx", "xls"])
    if uploaded_file is None:
        st.stop()

    # DataFrame へ読み込み
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Excelの読み込みに失敗しました: {e}")
        st.stop()

    st.write("アップロードされたデータ:")
    st.dataframe(df)
    
    # フリーテキスト列の自動検出と手動調整
    st.subheader("列タイプ自動分析・選択")
    
    # 自動検出実行
    detected_text_columns, analysis_results = auto_detect_text_columns(df)
    
    # 分析結果の表示
    st.write("**各列の自動判定結果:**")
    for col_name, result in analysis_results.items():
        icon = "📝" if result["type"] == "free_text" else "🏷️"
        color = "green" if result["type"] == "free_text" else "blue"
        st.markdown(f":{color}[{icon} {col_name}: **{result['type']}**] - {result['reason']}")
    
    # フリーテキスト列の選択（自動検出結果をデフォルトに）
    text_column_options = [f"列{i+1}: {col}" for i, col in enumerate(df.columns)]
    
    # デフォルト選択の決定
    default_selection = "なし"
    if len(detected_text_columns) > 0:
        # 最初に検出されたフリーテキスト列をデフォルトに
        default_index = detected_text_columns[0]
        default_selection = f"列{default_index+1}: {df.columns[default_index]}"
    
    st.write("**フリーテキスト列の選択:**")
    selected_text_columns = st.multiselect(
        "自動判定結果を確認して、フリーテキストとして使用する列を選択してください（複数選択可能）",
        text_column_options,
        default=[default_selection] if default_selection != "なし" else []
    )
    
    use_text_similarity = len(selected_text_columns) > 0
    text_column_indices = []
    
    if use_text_similarity:
        for selected_col in selected_text_columns:
            col_index = int(selected_col.split(":")[0].replace("列", "")) - 1
            text_column_indices.append(col_index)
            
            # 自動判定と異なる選択をした場合の確認
            if col_index not in detected_text_columns:
                st.warning(f"⚠️ 注意: {df.columns[col_index]} は自動判定では「タグ系」と判定されましたが、フリーテキストとして使用されます。")
            else:
                st.success(f"✅ 自動判定と一致: {df.columns[col_index]}")
        
        st.write(f"**使用するフリーテキスト列:** {[df.columns[i] for i in text_column_indices]}")
    else:
        st.info("ℹ️ フリーテキスト類似度は使用せず、タグ系カテゴリのみで処理します。")

    # データ処理開始
    cat1_data = df.iloc[:, 1]  # 2列目
    cat2_data = df.iloc[:, 2]  # 3列目
    cat3_data = df.iloc[:, 3]  # 4列目
    
    # フリーテキストデータ（複数列対応）
    text_data_list = []
    if use_text_similarity and text_column_indices:
        for col_idx in text_column_indices:
            text_data_list.append(df.iloc[:, col_idx])

    # ユニークキーを取得
    cat1_unique = cat1_data.dropna().unique().tolist()
    cat2_unique = cat2_data.dropna().unique().tolist()
    cat3_unique = cat3_data.dropna().unique().tolist()

    st.subheader("カテゴリごとのグループ設定")

    cat1_groups = group_selection(cat1_unique, label="カテゴリ1")
    cat2_groups = group_selection(cat2_unique, label="カテゴリ2")
    cat3_groups = group_selection(cat3_unique, label="カテゴリ3")
    
    # カテゴリごとの戦略設定
    st.subheader("📊 カテゴリごとの点数配分戦略")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        cat1_strategy = st.selectbox(
            "カテゴリ1の戦略",
            ["diversity", "homogeneity"],
            format_func=lambda x: "多様性重視（異なる値で高得点）" if x == "diversity" else "同質性重視（同じ値で高得点）"
        )
    with col2:
        cat2_strategy = st.selectbox(
            "カテゴリ2の戦略", 
            ["diversity", "homogeneity"],
            format_func=lambda x: "多様性重視（異なる値で高得点）" if x == "diversity" else "同質性重視（同じ値で高得点）"
        )
    with col3:
        cat3_strategy = st.selectbox(
            "カテゴリ3の戦略",
            ["diversity", "homogeneity"], 
            format_func=lambda x: "多様性重視（異なる値で高得点）" if x == "diversity" else "同質性重視（同じ値で高得点）"
        )

    # カラム追加: cat1_group, cat2_group, cat3_group
    df["cat1_group"] = assign_groups(cat1_data, cat1_groups)
    df["cat2_group"] = assign_groups(cat2_data, cat2_groups)
    df["cat3_group"] = assign_groups(cat3_data, cat3_groups)
    
    # テキスト類似度グループ機能は削除（高度グルーピングで直接処理）

    st.write("グループ分け結果 (catX_group列が追加されました):")
    st.dataframe(df)

    # 優先順位設定
    st.subheader("優先順位設定")
    
    # 基本の3カテゴリのみ（テキスト類似度は高度グルーピングで直接処理）
    priority_options = ["cat1_group", "cat2_group", "cat3_group"]
        
    st.write("利用可能な要素:")
    st.write("- cat1_group: カテゴリ1のグループ")
    st.write("- cat2_group: カテゴリ2のグループ") 
    st.write("- cat3_group: カテゴリ3のグループ")
    
    priorities = st.multiselect(
        "優先順位を設定してください (上から順に 7点, 5点, 3点, 1点)",
        priority_options,
        default=priority_options
    )
    
    # 点数マップ (7, 5, 3, 1点システム)
    score_map = {}
    score_values = [7, 5, 3, 1]
    for i, priority in enumerate(priorities):
        if i < len(score_values):
            score_map[priority] = score_values[i]

    st.write("優先順位:", priorities)
    st.write("点数マップ:", score_map)

    # 高度グルーピング設定
    st.subheader("🚀 高度グルーピング設定")
    
    # 高度グルーピングのみ使用
    use_advanced_grouping = True
    
    # テキスト列の選択（高度グルーピング用）
    st.write("**高度グルーピング用テキスト列設定:**")
        
    # 類似度計算モードの選択
    similarity_mode = st.radio(
        "類似度計算モードを選択してください",
        ["単一列での類似度", "2列結合での類似度"],
        help="単一列：1つの列内でのテキスト類似度 / 2列結合：2つの列を結合してテキスト類似度"
    )
    
    # テキスト列1の選択
    text_col1_options = [f"列{i+1}: {col}" for i, col in enumerate(df.columns)]
    selected_text_col1 = st.selectbox(
        "テキスト列1を選択してください（例：スキル情報）",
        text_col1_options,
        index=1 if len(df.columns) > 1 else 0
    )
    text_col1_index = int(selected_text_col1.split(":")[0].replace("列", "")) - 1
    
    # テキスト列2の選択（2列結合モードの場合のみ）
    text_col2_index = None
    if similarity_mode == "2列結合での類似度":
        selected_text_col2 = st.selectbox(
            "テキスト列2を選択してください（例：自己PR）",
            text_col1_options,
            index=2 if len(df.columns) > 2 else 0
        )
        text_col2_index = int(selected_text_col2.split(":")[0].replace("列", "")) - 1
    
    # テキスト類似度の重み設定
    st.write("**重み設定:**")
    w_sim = st.slider("テキスト類似度の重み", 0, 10, 1, 1)
    
    # 固定グループサイズ
    target_group_size = st.number_input("希望グループサイズ", min_value=2, value=4, step=1)
    
    st.write(f"**設定確認:**")
    st.write(f"- 類似度計算モード: {similarity_mode}")
    st.write(f"- テキスト列1: {df.columns[text_col1_index]}")
    if text_col2_index is not None:
        st.write(f"- テキスト列2: {df.columns[text_col2_index]}")
    st.write(f"- テキスト類似度の重み: {w_sim}")


    # dfに固定フラグ, best_group_id, chunk_id がなければ作成
    if "fixed" not in df.columns:
        df["fixed"] = False  # 最高点満点で固定されたかどうか
    if "best_group_id" not in df.columns:
        df["best_group_id"] = np.nan
    if "chunk_id" not in df.columns:
        df["chunk_id"] = np.nan
    

    # 実行ボタン（高度グルーピングのみ）
    execute_button = st.button("🚀 高度グルーピング実行")
    
    if execute_button:
        # 高度グルーピング実行
            st.info("🚀 高度グルーピングを開始します...")
            
            # 類似度行列の作成
            similarity_matrix = create_advanced_similarity_matrix(
                df, text_col1_index, text_col2_index, client, DEPLOYMENT_NAME
            )
            
            if similarity_matrix is not None:
                # 高度グルーピング実行
                # カテゴリ戦略辞書を作成
                category_strategies = {
                    1: cat1_strategy,  # cat1_group列用の戦略
                    2: cat2_strategy,  # cat2_group列用の戦略 
                    3: cat3_strategy   # cat3_group列用の戦略
                }
                
                advanced_groups = find_advanced_optimal_groups(
                    df, similarity_matrix, priority_options, score_map,
                    category_strategies, target_group_size, w_sim # min_group_size -> target_group_size に修正
                )
                
                if advanced_groups:
                    # 結果の表示
                    display_advanced_group_analysis(df, advanced_groups, priority_options, score_map, category_strategies)
                    
                    # 結果DataFrameの作成
                    advanced_result_df = create_advanced_results_dataframe(df, advanced_groups)
                    if advanced_result_df is not None:
                        st.subheader("📊 高度グルーピング結果")
                        st.dataframe(advanced_result_df)
                        
                        # CSV ダウンロード
                        csv_data = advanced_result_df.to_csv(index=False, encoding="shift_jis")
                        st.download_button(
                            "📥 高度グルーピング結果CSV (Shift-JIS)",
                            data=csv_data.encode("shift_jis"),
                            file_name="advanced_grouping_results.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("高度グルーピングに失敗しました。")
            else:
                st.error("類似度行列の作成に失敗しました。従来方式を試してください。")
        
if __name__ == "__main__":
    main()