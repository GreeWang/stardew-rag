# evaluation_set_creator.py
import json
import random
import os
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvaluationSetCreator:
    def __init__(self, base_path: str = "data"):
        self.base_path = base_path
        self.evaluation_set = []
    
    def load_original_data(self) -> List[Dict]:
        """加载原始JSON数据"""
        data_path = os.path.join(self.base_path, "rag_docs.json")
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载 {len(data)} 条原始数据")
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"直接加载JSON失败: {e}，尝试逐行解析...")
            # 如果直接加载失败，尝试逐行解析
            return self._load_jsonl_data(data_path)
        except Exception as e:
            logger.error(f"加载原始数据失败: {e}")
            return []
    
    def _load_jsonl_data(self, file_path: str) -> List[Dict]:
        """处理JSONL格式或格式不正确的JSON文件"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 尝试解析每一行作为JSON对象
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析失败: {e}，跳过该行")
                        continue
                        
            logger.info(f"成功解析 {len(data)} 条数据")
            return data
        except Exception as e:
            logger.error(f"逐行解析文件失败: {e}")
            return []
    
    def clean_wiki_text(self, text: str) -> str:
        """
        清理wiki文本，只移除"来自Stardew Valley Wiki"固定句子
        
        Args:
            text: 原始wiki文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        cleaned_text = text
        
        # 只移除"来自Stardew Valley Wiki"及其变体
        patterns_to_remove = [
            r"来自Stardew Valley Wiki",
            r"来自.*?Stardew Valley Wiki",
        ]
        
        for pattern in patterns_to_remove:
            import re
            cleaned_text = re.sub(pattern, "", cleaned_text)
        
        # 清理多余的空格（但保留单个空格）
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def generate_question_from_item(self, item: Dict, index: int) -> Dict:
        """
        根据数据项生成评估问题
        
        Args:
            item: 原始数据项，包含title和text
            index: 项目索引
            
        Returns:
            包含问题、答案和来源的评估项
        """
        title = item.get("title", "")
        text = item.get("text", "")
        
        # 清理文本：移除固定的wiki标记
        cleaned_text = self.clean_wiki_text(text)
        
        # 从标题中提取主要名称（去掉"- 星露谷物语官方中文维基"部分）
        clean_title = title.split(" - ")[0] if " - " in title else title
        
        # 生成问题 - 使用"XX是什么？"格式
        question = f"{clean_title}是什么？"
        
        # 生成评估项
        evaluation_item = {
            "id": index + 1,
            "question": question,
            "true_answer": cleaned_text,  # 使用清理后的文本作为真实答案
            "true_sources": [clean_title],  # 使用标题作为来源
            "original_title": title,
            "difficulty": random.choice(["简单", "中等", "困难"])
        }
        
        return evaluation_item
    
    def create_evaluation_set(self, num_items: int = 60) -> List[Dict]:
        """
        从原始数据中随机选择条目创建评估集
        
        Args:
            num_items: 评估集大小
            
        Returns:
            评估集列表
        """
        # 加载原始数据
        original_data = self.load_original_data()
        
        if not original_data:
            logger.error("无法加载原始数据，无法创建评估集")
            return []
        
        # 随机选择指定数量的条目
        if len(original_data) < num_items:
            logger.warning(f"原始数据只有 {len(original_data)} 条，将使用所有数据")
            selected_items = original_data
        else:
            selected_items = random.sample(original_data, num_items)
        
        # 生成评估集
        self.evaluation_set = []
        for i, item in enumerate(selected_items):
            evaluation_item = self.generate_question_from_item(item, i)
            self.evaluation_set.append(evaluation_item)
        
        # 保存评估集
        eval_path = os.path.join(self.base_path, "evaluation_set.json")
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_set, f, ensure_ascii=False, indent=2)
        
        logger.info(f"成功创建包含 {len(self.evaluation_set)} 个问题的评估集")
        return self.evaluation_set
    
    def print_evaluation_set_sample(self, num_samples: int = 5):
        """打印评估集样本"""
        if not self.evaluation_set:
            logger.error("评估集为空，请先创建评估集")
            return
        
        print(f"\n=== 评估集样本 (前{num_samples}个) ===")
        for i, item in enumerate(self.evaluation_set[:num_samples]):
            print(f"\n--- 问题 {i+1} ---")
            print(f"问题: {item['question']}")
            print(f"来源: {item['true_sources']}")
            print(f"难度: {item['difficulty']}")
            print(f"答案预览: {item['true_answer'][:100]}...")
        print("=" * 50)
    
    def get_evaluation_set_stats(self) -> Dict:
        """获取评估集统计信息"""
        if not self.evaluation_set:
            return {}
        
        difficulties = [item["difficulty"] for item in self.evaluation_set]
        difficulty_counts = {
            "简单": difficulties.count("简单"),
            "中等": difficulties.count("中等"),
            "困难": difficulties.count("困难")
        }
        
        return {
            "total_questions": len(self.evaluation_set),
            "difficulty_distribution": difficulty_counts
        }


def create_evaluation_set_main():
    """独立的创建评估集主函数"""
    creator = EvaluationSetCreator()
    
    print("开始创建评估集...")
    evaluation_set = creator.create_evaluation_set(50)
    
    if evaluation_set:
        print(f"\n成功创建包含 {len(evaluation_set)} 个问题的评估集")
        creator.print_evaluation_set_sample(5)
        
        # 显示评估集统计信息
        stats = creator.get_evaluation_set_stats()
        print(f"\n=== 评估集统计 ===")
        print(f"总问题数: {stats['total_questions']}")
        print(f"难度分布: {stats['difficulty_distribution']}")
    else:
        print("创建评估集失败")


if __name__ == "__main__":
    create_evaluation_set_main()