# document_processor.py
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import logging
import re

logger = logging.getLogger(__name__)

def clean_text(text):
    """
    Clean up invisible characters, remove bullets, symbols, and extra spacing.
    Keeps original newlines intact.
    """

    # Remove invisible/control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\u200B-\u200D\uFEFF]', '', text)

    # Remove common bullets and symbols
    text = re.sub(r'[•·▪●♦►☑★→←↓↑⇨∗※※➤➔➣✦✔✸⦿◆◇⬤]', '', text)

    # Remove any leftover ASCII/Unicode bullets or symbols
    text = re.sub(r'[^\w\s\n.,;:!?@#%&()\'"/\-]', '', text)

    # Insert newline if jammed uppercase (e.g., TeamworkInnovationLANGUAGES)
    text = re.sub(r'(?<=[a-z])(?=[A-Z]{3,})', '\n', text)

    # Normalize multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    return text


class CVProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200, single_chunk=True, save_txt=True, txt_output_dir="data/txt_cvs"):
        """
        Initialize CVProcessor
       
        Args:
            chunk_size: Size of chunks when splitting (ignored if single_chunk=True)
            chunk_overlap: Overlap between chunks (ignored if single_chunk=True)
            single_chunk: If True, keeps entire CV as one chunk
            save_txt: If True, saves processed CV content as text files
            txt_output_dir: Directory to save text files
        """
        self.single_chunk = single_chunk
        self.save_txt = save_txt
        self.txt_output_dir = txt_output_dir
        
        # Create output directory if it doesn't exist
        if self.save_txt:
            os.makedirs(self.txt_output_dir, exist_ok=True)
            logger.info(f"Text output directory: {self.txt_output_dir}")
       
        if not single_chunk:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
                length_function=len
            )
    
    def _get_output_filename(self, file_path):
        """Generate output filename for text file"""
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        return os.path.join(self.txt_output_dir, f"{base_name}.txt")
    
    def _save_as_text(self, content, output_path):
        """Save content as text file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved text file: {output_path}")
        except Exception as e:
            logger.error(f"Error saving text file {output_path}: {e}")
            raise
   
    def process_cv(self, file_path):
        """Process a single CV file and return document chunks"""
        try:
            # Detect file type and use appropriate loader
            file_extension = os.path.splitext(file_path)[1].lower()
           
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_extension in ['.txt', '.text']:
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
           
            # Load documents
            documents = loader.load()
           
            if self.single_chunk:
                # Combine all pages/sections into a single document
                combined_content = ""
                combined_metadata = {}
               
                for doc in documents:
                    combined_content += doc.page_content + "\n"
                    # Merge metadata from all pages
                    combined_metadata.update(doc.metadata)
               
                # Clean up the combined content (keep newlines)
                combined_content = clean_text(combined_content)
                
                # Save as text file if enabled
                if self.save_txt:
                    output_path = self._get_output_filename(file_path)
                    self._save_as_text(combined_content, output_path)
               
                # Create single document with combined content
                single_doc = Document(
                    page_content=combined_content,
                    metadata={
                        **combined_metadata,
                        'source_file': file_path,
                        'txt_output': output_path if self.save_txt else None
                    }
                )
               
                chunks = [single_doc]
                logger.info(f"Processed {file_path}: Single chunk created with {len(combined_content)} characters")
            else:
                # Split documents as before
                chunks = self.text_splitter.split_documents(documents)
                
                # If saving text files, combine all chunks for the text file
                if self.save_txt:
                    combined_content = "\n".join([chunk.page_content for chunk in chunks])
                    combined_content = clean_text(combined_content)
                    output_path = self._get_output_filename(file_path)
                    self._save_as_text(combined_content, output_path)
                    
                    # Add txt_output path to metadata of all chunks
                    for chunk in chunks:
                        chunk.metadata['txt_output'] = output_path
                
                logger.info(f"Processed {file_path}: {len(chunks)} chunks created")
           
            return chunks
           
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            raise
   
    def process_multiple_cvs(self, file_paths):
        """Process multiple CV files"""
        all_chunks = []
        processed_files = []
        failed_files = []
       
        for file_path in file_paths:
            try:
                chunks = self.process_cv(file_path)
                all_chunks.extend(chunks)
                processed_files.append(file_path)
            except Exception as e:
                logger.warning(f"Skipping {file_path} due to error: {e}")
                failed_files.append((file_path, str(e)))
                continue
        
        # Log summary
        logger.info(f"Processing complete: {len(processed_files)} files processed successfully, {len(failed_files)} failed")
        if failed_files:
            logger.warning(f"Failed files: {[f[0] for f in failed_files]}")
       
        return all_chunks
    
    def get_txt_files_info(self):
        """Get information about saved text files"""
        if not self.save_txt or not os.path.exists(self.txt_output_dir):
            return []
        
        txt_files = []
        for filename in os.listdir(self.txt_output_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.txt_output_dir, filename)
                file_size = os.path.getsize(file_path)
                txt_files.append({
                    'filename': filename,
                    'path': file_path,
                    'size_bytes': file_size
                })
        
        return txt_files
