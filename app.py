import os
import sys
import logging
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import json
from typing import Dict, Any, List, Optional, Tuple
import webbrowser 
import datetime  


from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine
import web_app  
from config import (
    LLM_PROVIDERS,
    VECTOR_DB_CONFIG,
    DOCUMENT_PROCESSOR_CONFIG,
    LLM_CONFIG,
    RAG_CONFIG,
    APP_CONFIG
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGApplication(tk.Tk):
    def __init__(self):
        super().__init__()

        # Initialize configuration
        self.app_config = APP_CONFIG
        self.vector_config = VECTOR_DB_CONFIG.copy()
        self.processor_config = DOCUMENT_PROCESSOR_CONFIG.copy()
        self.llm_config = LLM_CONFIG.copy()
        self.rag_config = RAG_CONFIG.copy()

        # Initialize components
        self.document_processor = DocumentProcessor(self.processor_config)
        self.vector_store = VectorStore(self.vector_config)
        self.rag_engine = RAGEngine(
            self.vector_store,
            self.rag_config,
            self.llm_config
        )

        # Added: Used to track Web server thread
        self.web_server_thread = None

        # Configure interface
        self.title(self.app_config["app_title"])
        self.geometry(f"{self.app_config['app_width']}x{self.app_config['app_height']}")

        # Create main frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tab control
        self.tab_control = ttk.Notebook(self.main_frame)

        # Create tabs
        self.vector_store_tab = ttk.Frame(self.tab_control)
        

        self.tab_control.add(self.vector_store_tab, text='Vector Knowledge Base Management')
       

        self.tab_control.pack(expand=1, fill=tk.BOTH)

        # Initialize vector knowledge base management page
        self._init_vector_store_tab()
        

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Update status
        self._update_status()

        # Added: Handle window closing event
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _init_vector_store_tab(self):
        """Initialize vector knowledge base management tab"""
        # Create left and right split layout
        left_frame = ttk.Frame(self.vector_store_tab)
        right_frame = ttk.Frame(self.vector_store_tab)

        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left side: File upload and management
        ttk.Label(left_frame, text="Document Management", font=("Arial", 12, "bold")).pack(pady=10)

        # File upload area
        upload_frame = ttk.LabelFrame(left_frame, text="Document Upload")
        upload_frame.pack(fill=tk.X, padx=5, pady=5)

        # Single file upload
        ttk.Label(upload_frame, text="Upload Single File:").pack(anchor=tk.W, padx=5, pady=2)
        file_frame = ttk.Frame(upload_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=2)

        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="Browse", command=self._browse_file).pack(side=tk.RIGHT, padx=5)
        ttk.Button(upload_frame, text="Upload File", command=self._upload_file).pack(anchor=tk.E, padx=5, pady=5)

        # Folder upload
        ttk.Label(upload_frame, text="Upload Folder:").pack(anchor=tk.W, padx=5, pady=2)
        dir_frame = ttk.Frame(upload_frame)
        dir_frame.pack(fill=tk.X, padx=5, pady=2)

        self.dir_path_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.dir_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dir_frame, text="Browse", command=self._browse_directory).pack(side=tk.RIGHT, padx=5)
        ttk.Button(upload_frame, text="Upload Folder", command=self._upload_directory).pack(anchor=tk.E, padx=5, pady=5)

        # File management area
        files_frame = ttk.LabelFrame(left_frame, text="Indexed Files")
        files_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # File list
        self.files_listbox = tk.Listbox(files_frame)
        self.files_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # File management buttons
        files_btn_frame = ttk.Frame(files_frame)
        files_btn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(files_btn_frame, text="Refresh List", command=self._refresh_file_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(files_btn_frame, text="Delete Selected", command=self._delete_selected_file).pack(side=tk.RIGHT, padx=5)

        # Right side: Vector store parameter settings
        ttk.Label(right_frame, text="Vector Store Config", font=("Arial", 12, "bold")).pack(pady=10)

        params_frame = ttk.LabelFrame(right_frame, text="Parameter Settings")
        params_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Vector store path
        path_frame = ttk.Frame(params_frame)
        path_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(path_frame, text="Vector Store Path:").pack(side=tk.LEFT)
        self.vs_path_var = tk.StringVar(value=self.vector_config["vector_store_path"])
        ttk.Entry(path_frame, textvariable=self.vs_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(path_frame, text="Browse", command=self._browse_vector_store_path).pack(side=tk.RIGHT)

        # Embedding model
        model_frame = ttk.Frame(params_frame)
        model_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(model_frame, text="Embedding Model:").pack(side=tk.LEFT)
        self.embedding_model_var = tk.StringVar(value=self.vector_config["embedding_model"])
        ttk.Combobox(model_frame, textvariable=self.embedding_model_var, values=[
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Distance metric
        metric_frame = ttk.Frame(params_frame)
        metric_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(metric_frame, text="Distance Metric:").pack(side=tk.LEFT)
        self.distance_metric_var = tk.StringVar(value=self.vector_config["distance_metric"])
        ttk.Combobox(metric_frame, textvariable=self.distance_metric_var, values=[
            "cosine", "euclidean", "dot_product"
        ]).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Number of results to return
        topk_frame = ttk.Frame(params_frame)
        topk_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(topk_frame, text="Top K Results:").pack(side=tk.LEFT)
        self.top_k_var = tk.StringVar(value=str(self.vector_config["top_k_results"]))
        ttk.Spinbox(topk_frame, from_=1, to=20, textvariable=self.top_k_var).pack(side=tk.LEFT, fill=tk.X, expand=True,
                                                                                  padx=5)

        # GPU usage option
        gpu_frame = ttk.Frame(params_frame)
        gpu_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(gpu_frame, text="Use GPU:").pack(side=tk.LEFT)
        self.use_gpu_var = tk.BooleanVar(value=self.vector_config.get("use_gpu", True))
        ttk.Checkbutton(gpu_frame, variable=self.use_gpu_var).pack(side=tk.LEFT, padx=5)

        # Batch size
        batch_frame = ttk.Frame(params_frame)
        batch_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(batch_frame, text="Batch Size:").pack(side=tk.LEFT)
        self.batch_size_var = tk.StringVar(value=str(self.vector_config.get("batch_size", 32)))
        ttk.Spinbox(batch_frame, from_=1, to=128, textvariable=self.batch_size_var).pack(side=tk.LEFT, fill=tk.X,
                                                                                         expand=True, padx=5)

        # Document processing parameters
        doc_frame = ttk.LabelFrame(right_frame, text="Document Processing Parameters")
        doc_frame.pack(fill=tk.X, padx=5, pady=5)

        # Chunk size
        chunk_size_frame = ttk.Frame(doc_frame)
        chunk_size_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(chunk_size_frame, text="Chunk Size:").pack(side=tk.LEFT)
        self.chunk_size_var = tk.StringVar(value=str(self.processor_config["chunk_size"]))
        ttk.Spinbox(chunk_size_frame, from_=100, to=5000, increment=100, textvariable=self.chunk_size_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Chunk overlap
        chunk_overlap_frame = ttk.Frame(doc_frame)
        chunk_overlap_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(chunk_overlap_frame, text="Chunk Overlap:").pack(side=tk.LEFT)
        self.chunk_overlap_var = tk.StringVar(value=str(self.processor_config["chunk_overlap"]))
        ttk.Spinbox(chunk_overlap_frame, from_=0, to=1000, increment=50, textvariable=self.chunk_overlap_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Max file size
        max_size_frame = ttk.Frame(doc_frame)
        max_size_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(max_size_frame, text="Max File Size (MB):").pack(side=tk.LEFT)
        self.max_file_size_var = tk.StringVar(value=str(self.processor_config["max_file_size_mb"]))
        ttk.Spinbox(max_size_frame, from_=1, to=200, textvariable=self.max_file_size_var).pack(side=tk.LEFT, fill=tk.X,
                                                                                               expand=True, padx=5)

        # Button area
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(anchor=tk.E, padx=5, pady=10)

        ttk.Button(button_frame, text="Save Config", command=self._save_vector_store_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Open Web UI", command=self._open_web_ui).pack(side=tk.LEFT, padx=5)

        # Initialize file list
        self._refresh_file_list()

    def _init_rag_settings_tab(self):
        """Initialize RAG settings tab"""
        # Create left and right split layout
        left_frame = ttk.Frame(self.rag_settings_tab)
        right_frame = ttk.Frame(self.rag_settings_tab)

        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left side: RAG configuration
        ttk.Label(left_frame, text="RAG Configuration", font=("Arial", 12, "bold")).pack(pady=10)

        rag_frame = ttk.LabelFrame(left_frame, text="Retrieval Augmentation Parameters")
        rag_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Similarity threshold
        similarity_frame = ttk.Frame(rag_frame)
        similarity_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(similarity_frame, text="Similarity Threshold:").pack(side=tk.LEFT)
        self.similarity_threshold_var = tk.StringVar(value=str(self.rag_config["similarity_threshold"]))
        ttk.Scale(similarity_frame, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.similarity_threshold_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(similarity_frame, textvariable=self.similarity_threshold_var).pack(side=tk.RIGHT, padx=5)

        # Max context length
        context_length_frame = ttk.Frame(rag_frame)
        context_length_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(context_length_frame, text="Max Context Length:").pack(side=tk.LEFT)
        self.context_length_var = tk.StringVar(value=str(self.rag_config["max_context_length"]))
        ttk.Spinbox(context_length_frame, from_=1000, to=10000, increment=500,
                    textvariable=self.context_length_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Context header
        context_header_frame = ttk.Frame(rag_frame)
        context_header_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(context_header_frame, text="Context Header:").pack(anchor=tk.W)
        self.context_header_var = tk.StringVar(value=self.rag_config["context_header"])
        ttk.Entry(context_header_frame, textvariable=self.context_header_var).pack(fill=tk.X, expand=True, padx=5,
                                                                                   pady=2)

        # Context footer
        context_footer_frame = ttk.Frame(rag_frame)
        context_footer_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(context_footer_frame, text="Context Footer:").pack(anchor=tk.W)
        self.context_footer_var = tk.StringVar(value=self.rag_config["context_footer"])
        ttk.Entry(context_footer_frame, textvariable=self.context_footer_var).pack(fill=tk.X, expand=True, padx=5,
                                                                                   pady=2)

        # Save button
        ttk.Button(left_frame, text="Save RAG Config", command=self._save_rag_config).pack(anchor=tk.E, padx=5, pady=10)

        # Right side: LLM configuration
        ttk.Label(right_frame, text="LLM Configuration", font=("Arial", 12, "bold")).pack(pady=10)

        llm_frame = ttk.LabelFrame(right_frame, text="LLM Parameters")
        llm_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


        # Model Provider
        provider_frame = ttk.Frame(llm_frame)
        provider_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(provider_frame, text="Model Provider:").pack(side=tk.LEFT)
        self.provider_var = tk.StringVar(value=self.llm_config["provider"])
        provider_combobox = ttk.Combobox(provider_frame, textvariable=self.provider_var,
                                         values=list(LLM_PROVIDERS.keys()))
        provider_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        provider_combobox.bind("<<ComboboxSelected>>", self._update_model_list)

        # Model Name
        model_name_frame = ttk.Frame(llm_frame)
        model_name_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(model_name_frame, text="Model Name:").pack(side=tk.LEFT)
        self.model_name_var = tk.StringVar(value=self.llm_config["model_name"])
        self.model_combobox = ttk.Combobox(model_name_frame, textvariable=self.model_name_var)
        self.model_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Initialize model list
        self._update_model_list()

        # Temperature
        temperature_frame = ttk.Frame(llm_frame)
        temperature_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(temperature_frame, text="Temperature:").pack(side=tk.LEFT)
        self.temperature_var = tk.StringVar(value=str(self.llm_config["temperature"]))
        ttk.Scale(temperature_frame, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.temperature_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(temperature_frame, textvariable=self.temperature_var).pack(side=tk.RIGHT, padx=5)

        # Max Output Tokens
        max_tokens_frame = ttk.Frame(llm_frame)
        max_tokens_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(max_tokens_frame, text="Max Output Tokens:").pack(side=tk.LEFT)
        self.max_tokens_var = tk.StringVar(value=str(self.llm_config["max_tokens"]))
        ttk.Spinbox(max_tokens_frame, from_=100, to=4000, increment=100, textvariable=self.max_tokens_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # OpenAI API Key
        openai_api_key_frame = ttk.Frame(llm_frame)
        openai_api_key_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(openai_api_key_frame, text="OpenAI API Key:").pack(side=tk.LEFT)
        self.openai_api_key_var = tk.StringVar(value=self.llm_config.get("openai_api_key", ""))
        ttk.Entry(openai_api_key_frame, textvariable=self.openai_api_key_var, show="*").pack(side=tk.LEFT, fill=tk.X,
                                                                                             expand=True, padx=5)

        # DeepSeek API Key
        deepseek_api_key_frame = ttk.Frame(llm_frame)
        deepseek_api_key_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(deepseek_api_key_frame, text="DeepSeek API Key:").pack(side=tk.LEFT)
        self.deepseek_api_key_var = tk.StringVar(value=self.llm_config.get("deepseek_api_key", ""))
        ttk.Entry(deepseek_api_key_frame, textvariable=self.deepseek_api_key_var, show="*").pack(side=tk.LEFT,
                                                                                                 fill=tk.X, expand=True,
                                                                                                 padx=5)

        # System Prompt
        system_prompt_frame = ttk.Frame(llm_frame)
        system_prompt_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(system_prompt_frame, text="System Prompt:").pack(anchor=tk.W)
        self.system_prompt_text = scrolledtext.ScrolledText(system_prompt_frame, height=5)
        self.system_prompt_text.pack(fill=tk.X, expand=True, padx=5, pady=2)
        self.system_prompt_text.insert(tk.END, self.llm_config["system_prompt"])

        # Save button
        ttk.Button(right_frame, text="Save LLM Config", command=self._save_llm_config).pack(anchor=tk.E, padx=5, pady=10)

    def _update_model_list(self, event=None):
        """Update model list based on selected provider"""
        provider = self.provider_var.get()
        if provider in LLM_PROVIDERS:
            models = LLM_PROVIDERS[provider].get("models", [])
            self.model_combobox['values'] = models

            # If the currently selected model is not in the list, select the first one
            if self.model_name_var.get() not in models and models:
                self.model_name_var.set(models[0])

    def _init_chat_tab(self):
        """Initialize chat query tab"""
        # Create chat layout
        chat_frame = ttk.Frame(self.chat_tab)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Split chat area and context area
        chat_paned = ttk.PanedWindow(chat_frame, orient=tk.HORIZONTAL)
        chat_paned.pack(fill=tk.BOTH, expand=True)

        # Left side: Chat area
        left_chat_frame = ttk.Frame(chat_paned)
        chat_paned.add(left_chat_frame, weight=2)

        # Chat history
        history_frame = ttk.LabelFrame(left_chat_frame, text="Chat History")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.chat_history = scrolledtext.ScrolledText(history_frame)
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.chat_history.config(state=tk.DISABLED)

        # Input area
        input_frame = ttk.Frame(left_chat_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        self.chat_input = scrolledtext.ScrolledText(input_frame, height=4)
        self.chat_input.pack(fill=tk.X, padx=5, pady=5)
        self.chat_input.bind("<Return>", self._send_message)

        send_frame = ttk.Frame(input_frame)
        send_frame.pack(fill=tk.X, padx=5)

        # --- Modified part: Put buttons in a container ---
        button_container = ttk.Frame(send_frame)
        button_container.pack(side=tk.RIGHT, padx=5, pady=5)

        ttk.Button(button_container, text="Send", command=self._send_message).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_container, text="Open Web UI", command=self._open_web_ui).pack(side=tk.LEFT, padx=5)
        # --- Modification ends ---

        ttk.Button(send_frame, text="Clear History", command=self._clear_chat_history).pack(side=tk.LEFT, padx=5, pady=5)

        # Right: Context and retrieval results
        right_chat_frame = ttk.Frame(chat_paned)
        chat_paned.add(right_chat_frame, weight=1)

        context_frame = ttk.LabelFrame(right_chat_frame, text="Retrieval Context")
        context_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.context_text = scrolledtext.ScrolledText(context_frame)
        self.context_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.context_text.config(state=tk.DISABLED)

    def _browse_file(self):
        """Browse and select file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("PDF Files", "*.pdf")]
        )
        if file_path:
            self.file_path_var.set(file_path)

    def _browse_directory(self):
        """Browse and select directory"""
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.dir_path_var.set(dir_path)

    def _browse_vector_store_path(self):
        """Browse and select vector store path"""
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.vs_path_var.set(dir_path)

    def _upload_file(self):
        """Upload single file to vector store"""
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showwarning("Warning", "Please select a file first")
            return

        self.status_var.set("Processing file...")
        self.update_idletasks()

        # Process file in background thread
        threading.Thread(target=self._process_file, args=(file_path,)).start()

    def _upload_directory(self):
        """Upload all files in directory to vector store"""
        dir_path = self.dir_path_var.get()
        if not dir_path:
            messagebox.showwarning("Warning", "Please select a directory first")
            return

        self.status_var.set("Processing directory...")
        self.update_idletasks()

        # Process directory in background thread
        threading.Thread(target=self._process_directory, args=(dir_path,)).start()

    def _process_file(self, file_path):
        """Process single file and add to vector store"""
        try:
            if not self.document_processor.is_file_supported(file_path):
                self._show_status_message("Unsupported file format or file too large")
                return

            file_id, chunks = self.document_processor.process_pdf(file_path)
            self.vector_store.add_documents(chunks)

            self._show_status_message(f"File {os.path.basename(file_path)} processed")
            self._refresh_file_list()
        except Exception as e:
            self._show_status_message(f"Error processing file: {str(e)}")
            logger.error(f"Error processing file: {str(e)}", exc_info=True)

    def _process_directory(self, dir_path):
        """Process all files in directory and add to vector store"""
        try:
            result = self.document_processor.process_directory(dir_path)

            total_files = len(result)
            total_chunks = sum(len(chunks) for chunks in result.values())

            for file_id, chunks in result.items():
                self.vector_store.add_documents(chunks)

            self._show_status_message(f"Directory processed, added {total_files} files, {total_chunks} chunks")
            self._refresh_file_list()
        except Exception as e:
            self._show_status_message(f"Error processing directory: {str(e)}")
            logger.error(f"Error processing directory: {str(e)}", exc_info=True)

    def _refresh_file_list(self):
        """Refresh file list"""
        # Clear list
        self.files_listbox.delete(0, tk.END)

        # Collect file information
        files = {}
        for chunk_id, meta in self.vector_store.metadata.items():
            file_id = meta.get("file_id")
            if file_id and file_id not in files:
                files[file_id] = {
                    "file_name": meta.get("file_name", "Unknown File"),
                    "file_path": meta.get("file_path", "")
                }

        # Add to list
        for file_id, info in files.items():
            self.files_listbox.insert(tk.END, f"{info['file_name']} ({file_id})")

    def _delete_selected_file(self):
        """Delete selected file"""
        selection = self.files_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a file to delete")
            return

        selected_item = self.files_listbox.get(selection[0])
        # Extract file ID from string (format: filename (fileID))
        file_id = selected_item.split("(")[-1].strip(")")

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete file {selected_item}?"):
            num_deleted = self.vector_store.delete_by_file_id(file_id)
            self._show_status_message(f"Deleted {num_deleted} chunks related to the file")
            self._refresh_file_list()

    def _save_vector_store_config(self):
        """Save vector store configuration"""
        try:
            # Collect vector store configuration
            vector_config = {
                "vector_store_path": self.vs_path_var.get(),
                "embedding_model": self.embedding_model_var.get(),
                "distance_metric": self.distance_metric_var.get(),
                "top_k_results": int(self.top_k_var.get()),
                "use_gpu": self.use_gpu_var.get(),
                "batch_size": int(self.batch_size_var.get())
            }

            # Collect document processing configuration
            processor_config = {
                "chunk_size": int(self.chunk_size_var.get()),
                "chunk_overlap": int(self.chunk_overlap_var.get()),
                "max_file_size_mb": int(self.max_file_size_var.get())
            }

            # Update configuration
            self.vector_store.update_config(vector_config)
            self.document_processor.update_config(processor_config)

            # Update configuration copy
            self.vector_config.update(vector_config)
            self.processor_config.update(processor_config)

            self._show_status_message("Vector store config saved")
        except Exception as e:
            self._show_status_message(f"Error saving config: {str(e)}")
            logger.error(f"Error saving config: {str(e)}", exc_info=True)

    def _save_rag_config(self):
        """Save RAG configuration"""
        try:
            # Collect RAG configuration
            rag_config = {
                "similarity_threshold": float(self.similarity_threshold_var.get()),
                "max_context_length": int(self.context_length_var.get()),
                "context_header": self.context_header_var.get(),
                "context_footer": self.context_footer_var.get()
            }

            # Update configuration
            self.rag_engine.update_rag_config(rag_config)

            # Update configuration copy
            self.rag_config.update(rag_config)

            self._show_status_message("RAG config saved")
        except Exception as e:
            self._show_status_message(f"Error saving RAG config: {str(e)}")
            logger.error(f"Error saving RAG config: {str(e)}", exc_info=True)

    def _save_llm_config(self):
        """Save LLM configuration"""
        try:
            # Collect LLM configuration
            llm_config = {
                "provider": self.provider_var.get(),
                "model_name": self.model_name_var.get(),
                "temperature": float(self.temperature_var.get()),
                "max_tokens": int(self.max_tokens_var.get()),
                "openai_api_key": self.openai_api_key_var.get(),
                "deepseek_api_key": self.deepseek_api_key_var.get(),
                "system_prompt": self.system_prompt_text.get("1.0", tk.END).strip()
            }

            # Update configuration
            self.rag_engine.update_llm_config(llm_config)

            # Update configuration copy
            self.llm_config.update(llm_config)

            self._show_status_message("LLM config saved")
        except Exception as e:
            self._show_status_message(f"Error saving LLM config: {str(e)}")
            logger.error(f"Error saving LLM config: {str(e)}", exc_info=True)

    def _send_message(self, event=None):
        """Send message and get response"""
        # Get input
        query = self.chat_input.get("1.0", tk.END).strip()
        if not query:
            return

        # Clear input box
        self.chat_input.delete("1.0", tk.END)

        # Add user message to chat history
        self._add_message_to_history("User", query)

        # Set status
        self.status_var.set("Processing query...")
        self.update_idletasks()

        # Process query in background thread
        threading.Thread(target=self._process_query, args=(query,)).start()

        return "break"  # Prevent default behavior of Enter key

    def _process_query(self, query):
        """Process query and display result"""
        try:
            # Call RAG engine to process query
            result = self.rag_engine.query(query)

            # Add assistant response to chat history
            self._add_message_to_history("Assistant", result["answer"])

            # Update context area
            self._update_context_area(result["contexts"])

            self._show_status_message("Query processed")
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            self._add_message_to_history("System", error_message)
            self._show_status_message(error_message)
            logger.error(error_message, exc_info=True)

    def _add_message_to_history(self, sender, message):
        """Add message to chat history"""
        self.chat_history.config(state=tk.NORMAL)

        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        # Set different styles based on sender
        if sender == "User":
            self.chat_history.insert(tk.END, f"[{timestamp}] You: \n", "user")
            self.chat_history.insert(tk.END, f"{message}\n\n", "user_message")
        elif sender == "Assistant":
            self.chat_history.insert(tk.END, f"[{timestamp}] Assistant: \n", "assistant")
            self.chat_history.insert(tk.END, f"{message}\n\n", "assistant_message")
        else:
            self.chat_history.insert(tk.END, f"[{timestamp}] {sender}: \n", "system")
            self.chat_history.insert(tk.END, f"{message}\n\n", "system_message")

        # Set styles
        self.chat_history.tag_config("user", foreground="blue")
        self.chat_history.tag_config("assistant", foreground="green")
        self.chat_history.tag_config("system", foreground="red")

        # Scroll to bottom
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)

    def _clear_chat_history(self):
        """Clear chat history"""
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.delete("1.0", tk.END)
        self.chat_history.config(state=tk.DISABLED)

        self.context_text.config(state=tk.NORMAL)
        self.context_text.delete("1.0", tk.END)
        self.context_text.config(state=tk.DISABLED)

    def _update_context_area(self, contexts):
        """Update context area"""
        self.context_text.config(state=tk.NORMAL)
        self.context_text.delete("1.0", tk.END)

        if not contexts:
            self.context_text.insert(tk.END, "No relevant context found")
        else:
            for i, ctx in enumerate(contexts):
                self.context_text.insert(tk.END, f"Document {i + 1} (from {ctx['source']}, Score: {ctx['score']:.4f}):\n",
                                         "context_header")
                self.context_text.insert(tk.END, f"{ctx['text']}\n\n", "context_text")

        # Set styles
        self.context_text.tag_config("context_header", foreground="blue")

        self.context_text.config(state=tk.DISABLED)

    def _update_status(self):
        """Update status bar information"""
        try:
            stats = self.vector_store.get_stats()
            device_info = f"Device: {stats['device']}"
            self.status_var.set(
                f"Ready | Vector Store: {stats['total_chunks']} chunks, {stats['unique_files']} files | {device_info}")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")

    def _show_status_message(self, message):
        """Show message in status bar and update status"""
        self.status_var.set(message)
        self.update_idletasks()

        # Delay updating full status
        self.after(2000, self._update_status)

    # --- New method ---
    def _open_web_ui(self):
        """Start Web server and open in browser"""
        host = '127.0.0.1'
        port = 5000
        url = f"http://{host}:{port}"

        if self.web_server_thread and self.web_server_thread.is_alive():
            logger.info("Web server is already running.")
        else:
            self.status_var.set("Starting Web server...")
            self.update_idletasks()
            # Run server in background daemon thread, so it exits when main program exits
            server_thread = threading.Thread(
                target=web_app.run_web_server,
                args=(self.rag_engine, host, port),
                daemon=True  # Set as daemon thread
            )
            server_thread.start()
            self.web_server_thread = server_thread
            self.after(1000, lambda: self._show_status_message("Web server started"))

        # Open new tab in browser
        webbrowser.open_new_tab(url)

    def _on_closing(self):
        """Handle window closing event"""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            logger.info("Application closing.")
            self.destroy()
    # --- New method ends ---


if __name__ == "__main__":
    # Ensure fitz is imported in main program, although it is mainly used in other modules
    import fitz

    app = RAGApplication()
    app.mainloop()