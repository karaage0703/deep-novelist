import gradio as gr
from .novel_generator import generate_and_save_novel
from .ollama_connection import get_server_url, check_server_availability, get_available_models


def create_web_interface():
    """Gradioのウェブインターフェースを作成する"""
    with gr.Blocks(title="Deep Novelist") as interface:
        gr.Markdown("# 📚 Deep Novelist")
        gr.Markdown("AIを使って小説を生成します")

        with gr.Row():
            with gr.Column():
                description_input = gr.Textbox(
                    label="プロットの説明",
                    placeholder="物語のプロットを入力してください...",
                    value="主人公が異世界に転生し、魔法を使って冒険する物語",
                    lines=5,
                )
                style_input = gr.Textbox(
                    label="文体の指定（オプション）",
                    placeholder="例: ライトノベル調、文学的な文体、など",
                    lines=2,
                )
                ip_input = gr.Textbox(label="Ollamaサーバーのアドレス(Option)", value="", lines=1)

                # サーバーからモデル一覧を取得する関数
                def get_models(ip):
                    base_url = get_server_url(ip)
                    if check_server_availability(base_url):
                        return get_available_models(base_url)
                    return []

                # モデル選択用のドロップダウンを追加
                model_input = gr.Dropdown(
                    label="使用するモデル",
                    choices=get_models(None),  # 初期状態ではローカルサーバーをチェック
                    value=None,
                )

                # IPアドレスが変更されたときにモデル一覧を更新
                def update_models(ip):
                    return gr.update(choices=get_models(ip))

                ip_input.change(fn=update_models, inputs=[ip_input], outputs=[model_input])

                generate_btn = gr.Button("小説を生成", variant="primary")

            with gr.Column():
                novel_output = gr.Textbox(label="生成された小説", lines=20, show_copy_button=True)
                status_output = gr.Textbox(label="ステータス", lines=3)

        generate_btn.click(
            fn=generate_and_save_novel,
            inputs=[description_input, model_input, style_input, ip_input],
            outputs=[novel_output, status_output],
            api_name="generate_novel",
            show_progress=True,
        )

        gr.Markdown("""
        ## 使い方
        1. 必要に応じてOllamaサーバーのアドレスを入力
        2. 使用するモデルを選択
        3. プロットの説明を入力
        4. 必要に応じて文体を指定
        5. 「小説を生成」ボタンをクリック
        6. 生成された小説を確認（リアルタイムで表示されます）
        7. 必要に応じてコピーボタンでテキストをコピー

        ## 注意事項
        - Ollamaサーバーが実行中である必要があります
        - サーバーに必要なモデルがインストールされている必要があります
        - リモートサーバーを使用する場合は、サーバーのIPアドレスを入力してください
        """)

    return interface


def main():
    """メイン関数"""
    print("Deep Novelist を起動中...", flush=True)
    interface = create_web_interface()
    interface.queue()  # ストリーム出力のためにキューを有効化
    interface.launch(server_name="127.0.0.1", server_port=7861, share=True)


if __name__ == "__main__":
    main()
