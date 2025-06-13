import dash
from dash import dcc, html, Input, Output, State, callback_context
import base64
import os
import logging
from pathlib import Path

# Ensure necessary directories exist
os.makedirs("data/cvs", exist_ok=True)
os.makedirs("data/txt_cvs", exist_ok=True)

# Logging
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("docx").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import core modules
from Preprocessing.document_processor import CVProcessor
from Preprocessing.vector_store import CVVectorStore
from RAG.rag_engine import EnhancedRAGEngine

# Initialize components
try:
    processor = CVProcessor(single_chunk=True, save_txt=True, txt_output_dir="data/txt_cvs")
    vector_store = CVVectorStore()
    rag_engine = EnhancedRAGEngine(vector_store)
    logger.info("CVProcessor, CVVectorStore, and EnhancedRAGEngine initialized successfully.")
except Exception as e:
    logger.error(f"Initialization error: {e}")
    processor = vector_store = rag_engine = None

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Smart Recruiter Assistant", style={'textAlign': 'center'}),
    html.Hr(),

    html.H2("1. Upload CVs"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center'
        },
        multiple=True
    ),
    html.Div(id='output-upload-status'),
    html.Button('Process Uploaded CVs', id='process-cvs-button', n_clicks=0),
    html.Div(id='output-processing-status'),

    dcc.Store(id='uploaded-files-store', data=[]),

    html.Hr(),
    html.H2("2. Query Candidates"),
    dcc.Input(id='query-input', type='text', placeholder='Enter your query...', style={'width': '60%'}),
    html.Button('Find Top Candidates', id='find-top-button', n_clicks=0),
    html.Button('Get All Relevant Candidates', id='get-all-button', n_clicks=0),
    html.Div(id='output-query-results'),

    html.Hr(),
    html.H2("3. Get Candidate Info"),
    dcc.Dropdown(id='candidate-dropdown', options=[], placeholder="Select a candidate"),
    html.Button('Get Info', id='get-candidate-info-button', n_clicks=0),
    html.Div(id='output-candidate-info'),
])

def extract_skill_from_query(query: str) -> str:
    prefix = "Who has experience in "
    if query.startswith(prefix):
        return query[len(prefix):].rstrip("?").strip()
    return query

@app.callback(
    Output('output-upload-status', 'children'),
    Output('process-cvs-button', 'disabled'),
    Output('uploaded-files-store', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    prevent_initial_call=True
)
def upload_files(list_of_contents, list_of_names, list_of_dates):
    if not list_of_contents:
        return html.Div("No files uploaded."), True, []

    uploaded = []
    messages = []

    for content, name in zip(list_of_contents, list_of_names):
        try:
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            save_path = os.path.join("data/cvs", name)
            with open(save_path, 'wb') as f:
                f.write(decoded)
            messages.append(html.Div(f"✓ Uploaded: {name}", style={'color': 'green'}))
            uploaded.append(name)
        except Exception as e:
            logger.error(f"Upload error for {name}: {e}")
            messages.append(html.Div(f"✗ Failed: {name}", style={'color': 'red'}))

    return messages, False, uploaded

@app.callback(
    Output('output-processing-status', 'children'),
    Output('candidate-dropdown', 'options'),
    Input('process-cvs-button', 'n_clicks'),
    State('uploaded-files-store', 'data'),
    prevent_initial_call=True
)
def process_uploaded_files(n_clicks, uploaded_files):
    if processor is None or vector_store is None:
        return html.Div("System not ready."), []

    if not uploaded_files:
        return html.Div("No uploaded files to process."), []

    messages = []
    all_documents = []

    for filename in uploaded_files:
        file_path = os.path.join("data/cvs", filename)
        try:
            chunks = processor.process_cv(file_path)
            candidate_name = Path(filename).stem.replace('_CV', '').replace('_cv', '').replace('_', ' ').title()
            for chunk in chunks:
                chunk.metadata.update({
                    "candidate_name": candidate_name,
                    "file_path": file_path,
                    "file_type": Path(file_path).suffix,
                    "document_type": "complete_cv"
                })
            all_documents.extend(chunks)
            messages.append(html.Div(f"✓ Processed: {filename}", style={'color': 'green'}))
        except Exception as e:
            logger.error(f"Processing error for {filename}: {e}")
            messages.append(html.Div(f"✗ Failed: {filename}", style={'color': 'red'}))

    if not all_documents:
        return html.Div(["No CVs processed."]), []

    success_count, failed = vector_store.add_cvs(all_documents)
    candidates = vector_store.get_all_candidates()
    candidate_options = [{'label': name, 'value': name} for name in sorted(candidates)]

    messages.append(html.Div(f"✓ {success_count} CVs added. {len(candidates)} unique candidates found.", style={'fontWeight': 'bold'}))
    if failed:
        messages.append(html.Div(f"⚠ {len(failed)} documents failed to store.", style={'color': 'orange'}))

    return messages, candidate_options

@app.callback(
    Output('output-query-results', 'children'),
    Input('find-top-button', 'n_clicks'),
    Input('get-all-button', 'n_clicks'),
    State('query-input', 'value'),
    prevent_initial_call=True
)
def query_candidates(top_click, all_click, query_text):
    if not query_text:
        return "Enter a query."

    ctx = callback_context
    if not ctx.triggered or rag_engine is None:
        return "Error: Engine not ready."

    btn_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        if btn_id == 'find-top-button':
            result = rag_engine.find_top_candidates(query_text, top_k=5)
            return html.Pre(result)
        else:
            skill = extract_skill_from_query(query_text)
            if skill == query_text:
                return "Use: Who has experience in [skill]?"
            result = rag_engine.get_all_candidates_for_skill(skill)
            return html.Pre(result)
    except Exception as e:
        logger.error(f"Query error: {e}")
        return f"Error: {str(e)}"

@app.callback(
    Output('output-candidate-info', 'children'),
    Input('get-candidate-info-button', 'n_clicks'),
    State('candidate-dropdown', 'value'),
    prevent_initial_call=True
)
def get_candidate_info(n_clicks, name):
    if not name:
        return "Select a candidate."

    try:
        info = rag_engine.get_candidate_info(name)
        return html.Pre(info)
    except Exception as e:
        logger.error(f"Info error: {e}")
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, port=8050)
