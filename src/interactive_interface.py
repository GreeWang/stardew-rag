# interactive_query.py
# interactive_interface.py
import logging

logger = logging.getLogger(__name__)


def run_interactive_query(rag_system):
    """
    运行交互式查询界面
    
    Args:
        rag_system: 初始化的RAG系统实例
    """
    print("\n=== 星露谷物语 RAG 系统 ===")
    print("输入 '退出' 或 'quit' 来结束程序")
    print("=" * 30)
    
    while True:
        try:
            question = input("\n请输入你的问题: ").strip()
            
            if question.lower() in ['退出', 'quit', 'exit']:
                print("感谢使用星露谷物语 RAG 系统，再见！")
                break
            
            if not question:
                print("问题不能为空，请重新输入。")
                continue

            # 设置检索数量
            try:
                top_k_input = input("请输入检索数量 (默认 3): ").strip()
                top_k = int(top_k_input) if top_k_input else 3
            except ValueError:
                print("输入无效，使用默认值 3")
                top_k = 3

            print(f"\n正在查询: {question}")
            print(f"检索数量: {top_k}")
            print("-" * 50)
            
            # 执行查询
            answer = rag_system.query(question, top_k=top_k)
            
            print(f"\n回答: {answer}")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n\n程序被用户中断，再见！")
            break
        except Exception as e:
            print(f"\n查询过程中出现错误: {e}")
            print("请重新输入问题。")