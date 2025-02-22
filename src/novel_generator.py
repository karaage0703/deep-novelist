import sys
import traceback
from typing import Optional, Tuple
from ollama import Client
from .ollama_connection import check_server_availability, get_available_models


def clean_text(text: str) -> str:
    """生成されたテキストをクリーンアップする"""
    # コードブロックの削除
    if "```" in text:
        # コードブロックの開始と終了を探す
        start = text.find("```")
        if start != -1:
            # 言語指定がある場合はその行を飛ばす
            next_newline = text.find("\n", start)
            if next_newline != -1:
                text = text[next_newline + 1 :]
            # 終了マーカーを削除
            end = text.rfind("```")
            if end != -1:
                text = text[:end]

    # 先頭と末尾の空白を削除
    text = text.strip()

    # 改行の正規化
    lines = text.splitlines()
    cleaned_lines = [line.strip() for line in lines]

    return "\n".join(cleaned_lines)


class NovelGenerator:
    def __init__(self, base_url: str):
        """小説生成器を初期化する

        Args:
            base_url: OllamaサーバーのベースURL

        Raises:
            ConnectionError: サーバーに接続できない場合
        """
        if not check_server_availability(base_url):
            raise ConnectionError(f"Ollamaサーバー({base_url})に接続できません")

        self.client = Client(host=base_url)
        self.models = get_available_models(base_url)

    def generate_novel(self, description: str, model: str, style: Optional[str] = None) -> str:
        """小説を生成する

        Args:
            description: 小説のプロット説明
            model: 使用するモデル名
            style: 文体の指定（オプション）

        Returns:
            str: 生成された小説のテキスト

        Raises:
            ValueError: モデルが選択されていないか、利用できない場合
        """
        if not model:
            raise ValueError("モデルが選択されていません")
        if model not in self.models:
            raise ValueError(f"選択されたモデル '{model}' は利用できません")

        # スタイルの指定があれば追加
        style_prompt = f"\n文体: {style}" if style else ""

        prompt = f"""
以下の説明に基づいて小説を書いてください。
マークダウンの装飾は使用せず、純粋なテキストのみを返してください。

プロット:
{description}
{style_prompt}

要件:
- 適切な段落分けを行う
- 読みやすい文章にする
- 物語に一貫性を持たせる
- 登場人物の感情や心理描写を含める

注意:
- マークダウンの装飾は使用しないでください
- テキストのみを返してください
"""
        response = self.client.generate(model=model, prompt=prompt, stream=False)

        # 生成されたテキストをクリーンアップ
        return clean_text(response["response"])


def generate_and_save_novel(
    description: str, model: str, style: Optional[str] = None, ip: Optional[str] = None
) -> Tuple[str, str]:
    """小説を生成して保存する

    Args:
        description: 小説のプロット説明
        model: 使用するモデル名
        style: 文体の指定（オプション）
        ip: サーバーのIPアドレス（オプション）

    Returns:
        Tuple[str, str]: (生成された小説のテキスト, ステータスメッセージ)
    """
    try:
        from .ollama_connection import get_server_url

        base_url = get_server_url(ip)
        generator = NovelGenerator(base_url)

        # 小説の生成
        generated_text = generator.generate_novel(description, model, style)

        # 生成されたテキストを一時ファイルに保存
        temp_file = "generated_novel.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(generated_text)

        return (
            generated_text,
            f"小説の生成が完了しました！\n保存先: {temp_file}",
        )

    except Exception as e:
        error_msg = f"エラーが発生しました: {str(e)}"
        print(error_msg, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return "", error_msg
