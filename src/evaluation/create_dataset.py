from langsmith import Client
from langsmith.utils import LangSmithConflictError

def create_evaluation_dataset():
    client = Client()
    dataset_name = "image_processing_agent"
    
    # Just one test case
    test_cases = [
        {
            "request": "生成一张日落图片并添加'美丽的夜晚'文字",
            "expected_sequence": [
                "图像生成智能体：已生成新图像",
                "文本叠加智能体：已在图像上添加文字"
            ]
        }
    ]
    
    try:
        # Delete existing dataset if it exists
        existing_datasets = client.list_datasets()
        for dataset in existing_datasets:
            if dataset.name == dataset_name:
                client.delete_dataset(dataset_id=dataset.id)
                print(f"已删除现有数据集： {dataset_name}")
        
        # Create new dataset
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="多智能体图像处理系统的测试用例"
        )
        
        # Add examples to the dataset
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[{"request": case["request"]} for case in test_cases],
            outputs=[{"expected_sequence": case["expected_sequence"]} for case in test_cases]
        )
        
        print(f"已创建新数据集，包含 {len(test_cases)} 个示例")
        return dataset
        
    except Exception as e:
        print(f"创建数据集时出错： {e}")
        return None 