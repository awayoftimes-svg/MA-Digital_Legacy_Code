This is the code for a self createad clone of myself in order to see how feasiblie, acceptable and controllable a self constructed AI Avatar can be. 
The bot would be working with a valid API key!

# LuniBot – Quickstart
Setup:

in bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Trage in .env deinen OPENAI_API_KEY ein

To build the rag: run build_index.py , it will then appear in out_data

Info: Videos that were transcribed have been removed to keep the repository small (Videos would have been 30 + GB), I can provide them if requested.

- both RAW and manually labeled transcriptions are included 

- checkRAG.py is to have a basic check on the RAG's structure and functionality.

- rag_chat_sqlite.py is for local RAG testing through the terminal.

- API Key has been deleted from the .env file for privacy reasons
