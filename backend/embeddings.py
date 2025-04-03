from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from InstructorEmbedding import INSTRUCTOR
from llama_index.core import Settings
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding

# Instructor Embeddings
class InstructorEmbeddings(BaseEmbedding):
    _model: INSTRUCTOR = PrivateAttr()
    _instruction: str = PrivateAttr()

    def __init__(
        self,
        instructor_model_name: str = "hkunlp/instructor-base",
        instruction: str = "Represent this document for semantic search across academic, report, and structured data formats around page numbers. If the content is from a research paper or presentation (PDF), focus on extracting key technical concepts, arguments, and any referenced figures or slides. If the content is from a DOCX file, capture the logical flow, section headings, and paragraph summaries relevant for understanding the main points and conclusions.",
        **kwargs: Any,
    ) -> None:
        """
        Initializes the InstructorEmbeddings class.

        Args:
            instructor_model_name (str): The name of the Instructor model to use. Defaults to "hkunlp/instructor-base".
            instruction (str): The instruction to give to the Instructor model. Defaults to a string that asks the model to extract key technical concepts, arguments, and referenced figures or slides for PDFs and logical flow, section headings, and paragraph summaries for DOCX files.
            **kwargs: Additional keyword arguments to pass to the BaseEmbedding constructor.
        """
        super().__init__(**kwargs)
        self._model = INSTRUCTOR(instructor_model_name)
        self._instruction = instruction

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, query]])
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, text]])
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            [[self._instruction, text] for text in texts]
        )
        return embeddings

# # Instantiate InstructorEmbeddings
# embed_model = InstructorEmbeddings(embed_batch_size=2)
# Settings.embed_model = embed_model
# Settings.chunk_size = 512