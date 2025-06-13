# app.py
import dash
from dash import dcc, html, Input, Output, State, callback_context
import base64
import datetime
import io
import os
import logging
import json

# Ensure necessary directories exist for data and text CVs
if not os.path.exists("data/cvs"):
    os.makedirs("data/cvs")
if not os.path.exists("data/txt_cvs"):
    os.makedirs("data/txt_cvs")

# Suppress warnings from pypdf and python-docx
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("docx").setLevel(logging.ERROR)

# Configure logging for the app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(_name_)

# Import your core modules
# Assuming these files are in the same directory as app.py or accessible via PYTHONPATH
# For deployment, consider packaging these into a proper Python package or ensuring they are in the sys.path
from document_processor import CVProcessor
from vector_store import CVVectorStore
from rag_engine import EnhancedRAGEngine

# Initialize global components (ensure these are initialized once for the app lifecycle)
# For larger applications, consider using a database or more robust state management
# outside of global variables for scalability.
try:
    processor = CVProcessor(single_chunk=True, save_txt=True, txt_output_dir="data/txt_cvs")
    vector_store = CVVectorStore()
    rag_engine = EnhancedRAGEngine(vector_store)
    logger.info("CVProcessor, CVVectorStore, and EnhancedRAGEngine initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing core components: {e}")
    # Handle initialization errors, maybe display a message to the user
    processor = None
    vector_store = None
    rag_engine = None

app = dash.Dash(_name_)
server = app.server

app.layout = html.Div([
    html.H1("Smart Recruiter Assistant", style={'textAlign': 'center', 'color': '#2C3E50'}),
    html.Hr(),

    html.H2("1. Upload CVs", style={'color': '#34495E'}),
    html.P("Upload PDF, DOCX, or TXT files. The content will be processed and added to the knowledge base."),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px 0'
        },
        multiple=True
    ),
    html.Div(id='output-upload-status'),
    html.Button('Process Uploaded CVs', id='process-cvs-button', n_clicks=0,
                style={'backgroundColor': '#28A745', 'color': 'white', 'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'marginBottom': '20px'}),
    html.Div(id='output-processing-status'),
    html.Hr(),

    html.H2("2. Query Candidates", style={'color': '#34495E'}),
    dcc.Input(
        id='query-input',
        type='text',
        placeholder='e.g., Who has experience in Machine Learning?',
        style={'width': '80%', 'padding': '10px', 'marginRight': '10px', 'borderRadius': '5px', 'border': '1px solid #CCC'}
    ),
    html.Button('Find Top Candidates', id='find-top-button', n_clicks=0,
                style={'backgroundColor': '#007BFF', 'color': 'white', 'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'marginRight': '10px'}),
    html.Button('Get All Relevant Candidates', id='get-all-button', n_clicks=0,
                style={'backgroundColor': '#17A2B8', 'color': 'white', 'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
    html.Div(id='output-query-results', style={'marginTop': '20px', 'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace', 'backgroundColor': '#F8F9FA', 'padding': '15px', 'borderRadius': '5px', 'border': '1px solid #EEE'}),
    html.Hr(),

    html.H2("3. Get Candidate Info (by name)", style={'color': '#34495E'}),
    dcc.Dropdown(
        id='candidate-dropdown',
        options=[], # Options will be loaded dynamically
        placeholder="Select a candidate to get detailed info",
        style={'width': '80%', 'marginBottom': '10px'}
    ),
    html.Button('Get Info', id='get-candidate-info-button', n_clicks=0,
                style={'backgroundColor': '#6C757D', 'color': 'white', 'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
    html.Div(id='output-candidate-info', style={'marginTop': '20px', 'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace', 'backgroundColor': '#F8F9FA', 'padding': '15px', 'borderRadius': '5px', 'border': '1px solid #EEE'}),
])

def extract_skill_from_query(query: str) -> str:
    """Extract skill/topic from query like 'Who has experience in ... ?'"""
    prefix = "Who has experience in "
    if query.startswith(prefix):
        return query[len(prefix):].rstrip("?").strip()
    return query

# Callback to store uploaded file contents temporarily and update status
@app.callback(
    Output('output-upload-status', 'children'),
    Output('process-cvs-button', 'disabled'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    prevent_initial_call=True
)
def upload_files(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is None:
        return html.Div("No files uploaded yet."), True

    uploaded_files_info = []
    for content, name, date in zip(list_of_contents, list_of_names, list_of_dates):
        # Decode the content and save it
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        
        try:
            # Save the file to a temporary directory for processing
            save_path = os.path.join("data/cvs", name)
            with open(save_path, 'wb') as f:
                f.write(decoded)
            uploaded_files_info.append(f"'{name}' ({(os.path.getsize(save_path) / 1024):.2f} KB) uploaded successfully.")
        except Exception as e:
            uploaded_files_info.append(f"Error processing '{name}': {e}")
            logger.error(f"Error saving uploaded file {name}: {e}")

    return html.Div([
        html.Div(info) for info in uploaded_files_info
    ]), False # Enable process button after upload

# Callback to process uploaded CVs
@app.callback(
    Output('output-processing-status', 'children'),
    Output('candidate-dropdown', 'options'),
    Input('process-cvs-button', 'n_clicks'),
    prevent_initial_call=True
)
def process_cvs(n_clicks):
    if n_clicks is None or n_clicks == 0:
        return "", []

    if processor is None or vector_store is None:
        return html.Div("Error: Core components not initialized. Check server logs.", style={'color': 'red'}), []

    cv_directory = "data/cvs"
    cv_files = [os.path.join(cv_directory, f) for f in os.listdir(cv_directory) if f.lower().endswith(('.pdf', '.docx', '.doc', '.txt'))]

    if not cv_files:
        return html.Div("No CV files found in 'data/cvs' to process.", style={'color': 'orange'}), []

    processing_messages = []
    all_documents = []
    
    for cv_file in cv_files:
        try:
            chunks = processor.process_cv(cv_file)
            candidate_name = os.path.splitext(os.path.basename(cv_file))[0].replace('CV', '').replace('_cv', '').replace('', ' ').replace('-', ' ').title()
            for chunk in chunks:
                chunk.metadata.update({
                    "candidate_name": candidate_name,
                    "file_path": cv_file,
                    "file_type": os.path.splitext(cv_file)[1],
                    "document_type": "complete_cv"
                })
            all_documents.extend(chunks)
            processing_messages.append(f"'{os.path.basename(cv_file)}' processed and ready to add to vector store.")
        except Exception as e:
            processing_messages.append(f"Error processing '{os.path.basename(cv_file)}': {e}")
            logger.error(f"Error processing CV file {cv_file}: {e}")
            continue
    
    if not all_documents:
        return html.Div(children=[html.Div(msg) for msg in processing_messages] + [html.Div("No documents were successfully processed.")], style={'color': 'red'}), []

    try:
        vector_store.add_cvs(all_documents)
        candidates = vector_store.get_all_candidates()
        candidate_options = [{'label': name, 'value': name} for name in sorted(candidates)]
        processing_messages.append(f"Successfully added {len(all_documents)} chunks from {len(cv_files)} CVs to the vector store. Found {len(candidates)} unique candidates.")
        return html.Div(children=[html.Div(msg) for msg in processing_messages], style={'color': 'green'}), candidate_options
    except Exception as e:
        processing_messages.append(f"Error adding CVs to vector store: {e}")
        logger.error(f"Error adding CVs to vector store: {e}")
        return html.Div(children=[html.Div(msg) for msg in processing_messages], style={'color': 'red'}), []

# Callback for query buttons
@app.callback(
    Output('output-query-results', 'children'),
    Input('find-top-button', 'n_clicks'),
    Input('get-all-button', 'n_clicks'),
    State('query-input', 'value'),
    prevent_initial_call=True
)
def handle_query_buttons(top_n_clicks, all_relevant_clicks, query_text):
    if not query_text:
        return html.Div("Please enter a query.", style={'color': 'orange'})

    if rag_engine is None:
        return html.Div("Error: RAG engine not initialized. Check server logs.", style={'color': 'red'})

    triggered_id = callback_context.triggered_id

    if triggered_id == 'find-top-button':
        logger.info(f"Finding top candidates for query: {query_text}")
        try:
            top_candidates_result, detailed_info = rag_engine.find_top_candidates(query_text, top_k=5)
            # Format output clearly
            output_str = f"Query: {query_text}\n"
            output_str += "="*60 + "\n\n"
            output_str += "TOP 5 CANDIDATES RANKING:\n"
            output_str += "-"*40 + "\n"
            output_str += top_candidates_result
            return html.Pre(output_str)
        except Exception as e:
            logger.error(f"Error finding top candidates: {e}")
            return html.Div(f"Error finding top candidates: {e}", style={'color': 'red'})
    
    elif triggered_id == 'get-all-button':
        logger.info(f"Getting all relevant candidates for query: {query_text}")
        try:
            skill = extract_skill_from_query(query_text)
            if not skill or skill == query_text: # If skill extraction failed or no specific skill
                return html.Div("Please enter a query in the format 'Who has experience in [skill]?' for 'Get All Relevant Candidates'.", style={'color': 'orange'})
            
            all_relevant = rag_engine.get_all_candidates_for_skill(skill)
            
            output_str = f"Query: {query_text}\n"
            output_str += "="*60 + "\n\n"
            output_str += f"ALL RELEVANT CANDIDATES FOR SKILL: {skill.upper()}\n"
            output_str += "-"*40 + "\n"
            output_str += all_relevant
            return html.Pre(output_str)
        except Exception as e:
            logger.error(f"Error getting all relevant candidates: {e}")
            return html.Div(f"Error getting all relevant candidates: {e}", style={'color': 'red'})
    
    return ""

# Callback for getting candidate info by name
@app.callback(
    Output('output-candidate-info', 'children'),
    Input('get-candidate-info-button', 'n_clicks'),
    State('candidate-dropdown', 'value'),
    prevent_initial_call=True
)
def get_candidate_info(n_clicks, selected_candidate_name):
    if n_clicks is None or n_clicks == 0:
        return ""

    if not selected_candidate_name:
        return html.Div("Please select a candidate from the dropdown.", style={'color': 'orange'})

    if rag_engine is None:
        return html.Div("Error: RAG engine not initialized. Check server logs.", style={'color': 'red'})
    
    logger.info(f"Getting info for candidate: {selected_candidate_name}")
    try:
        candidate_info = rag_engine.get_candidate_info(selected_candidate_name)
        output_str = f"Detailed Information for: {selected_candidate_name.upper()}\n"
        output_str += "="*60 + "\n\n"
        output_str += candidate_info
        return html.Pre(output_str)
    except Exception as e:
        logger.error(f"Error getting candidate info for {selected_candidate_name}: {e}")
        return html.Div(f"Error getting candidate information: {e}", style={'color': 'red'})


if _name_ == '_main_':
    app.run_server(debug=True, port=8050)