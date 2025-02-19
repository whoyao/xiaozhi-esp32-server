from abc import ABC, abstractmethod
import requests
import numpy as np
from typing import List, Union
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

class EMBD(ABC):
    @abstractmethod
    def encode(self, data):        
        pass

class SentenceEMBD(EMBD):
    def __init__(self, config: dict):
        self.ollama_base_url = config.get("ollama_base_url", "http://localhost:11434")
        self.model_name = "EntropyYue/jina-embeddings-v2-base-zh"
        logger.bind(tag=TAG).info(f"Using Ollama embeddings with model: {self.model_name}")
        
    def encode(self, data: Union[str, List[str]]) -> np.ndarray:
        """获取文本的 embeddings

        Args:
            data: 单个文本字符串或文本列表

        Returns:
            numpy.ndarray: embeddings 向量或向量数组
        """
        if isinstance(data, str):
            data = [data]
            
        try:
            embeddings = []
            for text in data:
                response = requests.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                embedding = response.json()["embedding"]
                embeddings.append(embedding)
                
            return np.array(embeddings)
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error getting embeddings from Ollama: {e}")
            raise

def create_instance(class_name: str, *args, **kwargs) -> EMBD:
    """工厂方法创建embedding实例"""
    
    model_map = {
        "SentenceTransformer": SentenceEMBD,        
    }

    if cls := model_map.get(class_name):
        return cls(*args, **kwargs)
    raise ValueError(f"不支持的embedding类型: {class_name}")