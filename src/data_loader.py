# src/data_loader.py

import json
import logging
from typing import Generator, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _yield_doc(item: dict, line_num: int) -> Generator[Tuple[str, str], None, None]:
    title = item.get("title", "").strip()
    text_content = item.get("text", "").strip()
    if text_content:
        yield title, text_content
    else:
        logger.debug(f"跳过条目 (行 {line_num}): 内容为空")


def load_and_parse_jsonl(file_path: str) -> Generator[Tuple[str, str], None, None]:
    """
    加载文档文件，支持 JSON 数组或 JSONL（每行一个 JSON 对象）。

    Args:
        file_path: 输入文件路径。

    Yields:
        (title, text_content) 元组。
    """
    logger.info(f"开始加载和解析文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    if not content:
        logger.warning(f"文件为空: {file_path}")
        return

    # JSON 数组格式: [{...}, {...}]
    if content.startswith('['):
        try:
            items = json.loads(content)
            for i, item in enumerate(items, start=1):
                if isinstance(item, dict):
                    yield from _yield_doc(item, i)
            logger.info(f"成功解析 JSON 数组，共 {len(items)} 条: {file_path}")
            return
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 数组解析失败，回退到 JSONL 模式: {e}")

    # JSONL 格式: 每行一个 JSON 对象
    for line_num, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        try:
            item = json.loads(line)
            yield from _yield_doc(item, line_num)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析错误 (行 {line_num}): {e}")
            logger.error(f"错误行内容: {line[:100]}...")
        except Exception as e:
            logger.error(f"处理行 {line_num} 时发生未知错误: {e}")

    logger.info(f"成功解析完文件: {file_path}")
