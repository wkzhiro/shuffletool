import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.title("カテゴリ優先度付きシャッフル - グループサイズ分割 + 端数1つ + chunk_id対応")

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

    # # ▼ 必須列チェック (例では「カテゴリ１」「カテゴリ２」「カテゴリ３」 が必須)
    # required_cols = ["カテゴリ１", "カテゴリ２", "カテゴリ３"]
    # if not all(col in df.columns for col in required_cols):
    #     st.error("エクセルに『カテゴリ１』『カテゴリ２』『カテゴリ３』の列がありません。")
    #     st.stop()

    # -------------------------------
    # 1) マルチセレクトでカテゴリごとのグループ分け
    # -------------------------------

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
    
    cat1_data = df.iloc[:, 1]  # 2列目
    cat2_data = df.iloc[:, 2]  # 3列目
    cat3_data = df.iloc[:, 3]  # 4列目

    # ユニークキーを取得
    cat1_unique = cat1_data.dropna().unique().tolist()
    cat2_unique = cat2_data.dropna().unique().tolist()
    cat3_unique = cat3_data.dropna().unique().tolist()

    st.subheader("カテゴリごとのグループ設定")

    cat1_groups = group_selection(cat1_unique, label="カテゴリ1")
    cat2_groups = group_selection(cat2_unique, label="カテゴリ2")
    cat3_groups = group_selection(cat3_unique, label="カテゴリ3")

    # カラム追加: cat1_group, cat2_group, cat3_group
    df["cat1_group"] = assign_groups(cat1_data, cat1_groups)
    df["cat2_group"] = assign_groups(cat2_data, cat2_groups)
    df["cat3_group"] = assign_groups(cat3_data, cat3_groups)

    st.write("グループ分け結果 (catX_group列が追加されました):")
    st.dataframe(df)

    # -------------------------------
    # 2) カテゴリの優先順位設定
    # -------------------------------
    st.subheader("カテゴリ優先順位設定")
    priority_options = ["cat1_group", "cat2_group", "cat3_group"]
    priorities = st.multiselect(
        "ドラッグで順序変更 (上から順に 5点,3点,1点)",
        priority_options,
        default=priority_options
    )
    # 点数マップ (一番上 → 5, 次 → 3, 最後 → 1)
    score_map = {}
    if len(priorities) > 0:
        score_map[priorities[0]] = 5
    if len(priorities) > 1:
        score_map[priorities[1]] = 3
    if len(priorities) > 2:
        score_map[priorities[2]] = 1

    st.write("優先順位:", priorities, "→ 点数マップ:", score_map)

    # -------------------------------
    # 3) グループサイズ, シャッフル回数など設定
    # -------------------------------
    group_size = st.number_input("1班あたりの人数 (例:3)", min_value=1, value=3, step=1)
    max_iter = st.number_input("最大シャッフル回数", min_value=1, value=100, step=1)

    # dfに固定フラグ, best_group_id, chunk_id がなければ作成
    if "fixed" not in df.columns:
        df["fixed"] = False  # 9点満点(=3カテゴリ全一致)で固定されたかどうか
    if "best_group_id" not in df.columns:
        df["best_group_id"] = np.nan
    if "chunk_id" not in df.columns:
        df["chunk_id"] = np.nan

    best_score = 0

    # -------------------------------
    # 4) 「シャッフル開始」ボタン
    # -------------------------------
    if st.button("シャッフル開始"):

        # iterationごとに chunk_id を発行するためのカウンタ
        global_chunk_counter = 0

        for iteration in range(int(max_iter)):
            # st.write(f"### シャッフル {iteration+1} 回目")

            # 未固定ユーザーだけ抽出
            unlocked_df = df[df["fixed"] == False].copy()
            if len(unlocked_df) == 0:
                st.info("未固定ユーザーがいません。")
                break

            # シャッフル
            unlocked_df["orig_index"] = unlocked_df.index
            shuffled = unlocked_df.sample(frac=1).reset_index(drop=True)
            total_len = len(shuffled)
            total_score_this_iter = 0

            # 1) まず (total_len // group_size) 個の「フルサイズchunk」を作る
            num_full_chunks = total_len // group_size
            remainder = total_len % group_size

            chunk_start = 0
            # --- フルサイズチャンク ---
            for i in range(num_full_chunks):
                chunk_end = chunk_start + group_size
                chunk = shuffled.iloc[chunk_start:chunk_end]

                # chunk_idを発行
                global_chunk_counter += 1
                this_chunk_id = (iteration * 10000) + global_chunk_counter

                # スコア計算
                score = 0
                for col, pts in score_map.items():
                    if len(chunk[col].dropna().unique()) == 1:
                        score += pts

                total_score_this_iter += score

                # 9点 & 人数=group_size => 固定
                if score == 9 and len(chunk) == group_size:
                    locked_indices = chunk["orig_index"].tolist()
                    df.loc[locked_indices, "fixed"] = True
                    st.write("9点達成 (フルチャンク) → 固定:")
                    st.dataframe(chunk)

                # chunk_id 書き込み
                for row_i in chunk.index:
                    orig_idx = chunk.at[row_i, "orig_index"]
                    df.loc[orig_idx, "chunk_id"] = this_chunk_id

                chunk_start = chunk_end

            # 2) 端数を1つだけ作る ( leftover chunk )
            if remainder > 0:
                # chunk = leftover
                leftover_chunk = shuffled.iloc[chunk_start:]
                global_chunk_counter += 1
                leftover_chunk_id = (iteration * 10000) + global_chunk_counter

                # スコア計算(端数でも行う)
                score = 0
                for col, pts in score_map.items():
                    if len(leftover_chunk[col].dropna().unique()) == 1:
                        score += pts

                total_score_this_iter += score

                # leftover_chunk_id 書き込み
                for row_i in leftover_chunk.index:
                    orig_idx = leftover_chunk.at[row_i, "orig_index"]
                    df.loc[orig_idx, "chunk_id"] = leftover_chunk_id

            # --- 表示タイミング: 5回ごと(0,5,10...) or ベスト更新時 ---
            # まずスコア判定
            updated_best = False
            if total_score_this_iter > best_score:
                best_score = total_score_this_iter
                best_df = df.copy(deep=True)
                updated_best = True

            # 表示条件
            if (iteration % 10 == 0) or updated_best:
                st.write(f"#### シャッフル {iteration} 回目")
                st.write(f"合計スコア: {total_score_this_iter}")
                if updated_best:
                    st.write("ベストスコア更新 →", best_score)
                # 表示したい情報を追加
                st.write("固定済みユーザー数:", df["fixed"].sum())
                st.dataframe(df.head(10))  # 一部だけ表示する例

            # 全員固定チェック
            if df["fixed"].all():
                st.success(f"全員固定されました (iteration={iteration})")
                break

        st.success("シャッフル終了")

        # --- (4) ループ後、best_df に「ベスト時」の chunk_id 情報が入っている ---
        if best_df is not None:
            st.write("#### ベストスコア:", best_score)
            st.write("#### ベスト時の全割り当て結果(一部表示):")
            st.dataframe(best_df.head(30))
        else:
            st.write("ベストスコアの更新がありませんでした。")

        # -------------------------------
        # 最終的に chunk_id を 1 から始まる連番 best_group_id に変換
        # -------------------------------
        # --- best_df から best_group_id も生成 ---
        if best_df is not None:
            # chunk_id → best_group_id
            unique_chunks = sorted(best_df["chunk_id"].dropna().unique())
            chunk_map = {}
            c_id = 1
            for old_id in unique_chunks:
                chunk_map[old_id] = c_id
                c_id+=1
            best_df["best_group_id"] = best_df["chunk_id"].map(chunk_map)

            # NaN補填
            current_max = 0 if len(chunk_map)==0 else max(chunk_map.values())
            new_gid = current_max+1
            for idx in best_df.index:
                if pd.isna(best_df.at[idx, "best_group_id"]):
                    best_df.at[idx,"best_group_id"] = new_gid
                    new_gid+=1

            st.write("#### ベストスコア時の best_group_id 割り当て:")
            st.dataframe(best_df.head(30))

            # CSV ダウンロード
            download_df = best_df.drop(columns=["chunk_id"])
            download_df = download_df.sort_values(by="best_group_id", ascending=True)

            csv_data = download_df.to_csv(index=False, encoding="shift_jis")
            st.download_button(
                "ベストスコア割り当てCSV (Shift-JIS)",
                data=csv_data.encode("shift_jis"),
                file_name="best_assignment.csv",
                mime="text/csv"
            )
        else:
            st.write("ベスト割り当てがありません。")

if __name__=="__main__":
    main()