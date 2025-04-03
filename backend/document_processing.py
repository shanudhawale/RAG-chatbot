import os
import logging
from pathlib import Path
from llama_index.core import Document
import json
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
# from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.utils.export import generate_multimodal_pages
from backend.config import logger

pdf_pipeline_options = PdfPipelineOptions()
pdf_pipeline_options.generate_page_images = True

# Docling Document Converter intialized
doc_converter = (
        DocumentConverter(  
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.XLSX,
            ],  # whitelist formats, non-matching files are ignored.
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options
                ),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline, pipeline_options=pdf_pipeline_options
                )
            },
        )
    )

def process_documents(input_paths, output_dir):
    logger.info(f"Processing documents with input paths: {input_paths} and output dir: {output_dir}")
    # print("Input files in document processing")
    """
    Process a list of input files and convert them to a list of Document objects

    Args:
        input_paths (list[tuple]): A list of tuples containing the input file path and the actual document name
        output_dir (str): The output directory for the processed documents

    Returns:
        tuple[list[Document], str]: A tuple containing a list of Document objects and the document type
    """
    docs = []
    inputfiles = [Path(ip[0]) for ip in input_paths]
    temp_file_names = [ip[0].split('/')[-1] for ip in input_paths]
    actual_filenames = [ip[-1] for ip in input_paths]
    conv_results = doc_converter.convert_all(inputfiles)
    for res in conv_results:
        logger.info(f"Processing file: {res.input.file.name}")
        if res.input.file.name.split('.')[-1] in ['docx','txt']:
            doc = res.document.export_to_markdown() 
            document = Document(text= doc ,metadata={"document_type": "docx",
                                                     "document_name": res.input.file.name,
                                                     "actual_doc_name": actual_filenames[temp_file_names.index(f'{res.input.file.name}')],
                                                     })
            logger.info(f"Created document: {res.input.file.name}")
            docs.append(document)

        elif res.input.file.name.split('.')[-1] in ['xlsx']:
            doc = res.document.export_to_markdown()
            document = Document(text= doc ,metadata={"document_type": "xlsx",
                                                     "document_name": res.input.file.name,
                                                     "actual_doc_name":  actual_filenames[temp_file_names.index(f'{res.input.file.name}')],
                                                     })
            logger.info(f"Created document: {res.input.file.name}")
            docs.append(document)

        elif res.input.file.name.split('.')[-1] == "pdf":
            # print(">>>>>>>>>> Processing pdf")
            for (content_text,content_md,content_dt,page_cells,page_segments,page,) in generate_multimodal_pages(res):
                page_no = page.page_no + 1
                # print(page_no,">>>>>>>", content_text)
                page_image_filename = Path(f'/app/backend/data_images/{output_dir}/{res.input.file.name}-{page_no}.jpg')
                with page_image_filename.open("wb") as fp:
                    page.image.save(fp, format="JPEG")
                # print("saved image", page_no)
                document = Document(text = f"page number:{page_no}\n"+content_md, metadata = {"document_type": "pdf",
                                                                   "pdf_name": res.input.file.name,
                                                                   "actual_doc_name": actual_filenames[temp_file_names.index(f'{res.input.file.name}')],
                                                                   "page_num": page.page_no + 1,
                                                                   "image_path":f'/app/backend/data_images/{output_dir}/{res.input.file.name}-{page_no}.jpg',
                                                                   })
                logger.info(f"Created document: {res.input.file.name}")
                docs.append(document)

    return docs, document.metadata['document_type']