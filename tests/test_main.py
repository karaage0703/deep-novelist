import pytest
from unittest.mock import Mock, patch
from src.ollama_connection import (
    check_server_availability,
    get_server_url,
    get_available_models,
)
from src.novel_generator import NovelGenerator, clean_text


def test_get_server_url():
    """サーバーURL取得のテスト"""
    assert get_server_url() == "http://localhost:11434"
    assert get_server_url("192.168.1.1") == "http://192.168.1.1:11434"


@patch("requests.get")
def test_check_server_availability(mock_get):
    """サーバー可用性チェックのテスト"""
    # サーバーが利用可能な場合
    mock_get.return_value.status_code = 200
    assert check_server_availability("http://localhost:11434") is True

    # サーバーが利用不可能な場合
    mock_get.return_value.status_code = 404
    assert check_server_availability("http://localhost:11434") is False

    # 接続エラーの場合
    mock_get.side_effect = Exception("Connection error")
    assert check_server_availability("http://localhost:11434") is False


@patch("requests.get")
def test_get_available_models(mock_get):
    """利用可能なモデル取得のテスト"""
    # 正常なレスポンスの場合（リスト形式）
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = [{"name": "model1"}, {"name": "model2"}]
    models = get_available_models("http://localhost:11434")
    assert models == ["model1", "model2"]

    # 正常なレスポンスの場合（辞書形式）
    mock_get.return_value.json.return_value = {"models": [{"name": "model1"}, {"name": "model2"}]}
    models = get_available_models("http://localhost:11434")
    assert models == ["model1", "model2"]

    # エラーレスポンスの場合
    mock_get.return_value.status_code = 404
    models = get_available_models("http://localhost:11434")
    assert models == []


def test_clean_text():
    """テキストクリーンアップのテスト"""
    # マークダウンコードブロックの削除
    text = "```python\nprint('hello')\n```\n"
    assert clean_text(text) == "print('hello')"

    # 空白行の正規化
    text = "line1\n\n  line2  \nline3"
    assert clean_text(text) == "line1\nline2\nline3"

    # 前後の空白削除
    text = "  \n  text  \n  "
    assert clean_text(text) == "text"


@patch("src.novel_generator.Client")
@patch("src.novel_generator.check_server_availability")
@patch("src.novel_generator.get_available_models")
def test_novel_generator(mock_get_models, mock_check_server, mock_client):
    """小説生成器のテスト"""
    # サーバー接続のモック
    mock_check_server.return_value = True
    mock_get_models.return_value = ["model1", "model2"]
    mock_client_instance = Mock()
    mock_client.return_value = mock_client_instance
    mock_client_instance.generate.return_value = {"response": "Generated story"}

    # 生成器の初期化
    generator = NovelGenerator("http://localhost:11434")
    assert generator.models == ["model1", "model2"]

    # 小説生成
    novel = generator.generate_novel(description="Test plot", model="model1", style="Test style")
    assert novel == "Generated story"

    # エラーケース：無効なモデル
    with pytest.raises(ValueError):
        generator.generate_novel(description="Test plot", model="invalid_model")

    # エラーケース：モデル未選択
    with pytest.raises(ValueError):
        generator.generate_novel(description="Test plot", model="")


@patch("src.novel_generator.NovelGenerator")
def test_generate_and_save_novel(mock_generator_class, tmp_path):
    """小説生成と保存のテスト"""
    from src.novel_generator import generate_and_save_novel

    # モックの設定
    mock_generator = Mock()
    mock_generator_class.return_value = mock_generator
    mock_generator.generate_novel.return_value = "Generated novel"

    # 正常系のテスト
    with patch("builtins.open"):
        text, status = generate_and_save_novel(description="Test plot", model="model1", style="Test style")
        assert text == "Generated novel"
        assert "完了しました" in status

    # エラー系のテスト
    mock_generator.generate_novel.side_effect = Exception("Test error")
    text, status = generate_and_save_novel(description="Test plot", model="model1")
    assert text == ""
    assert "エラー" in status
