import streamlit as st
import google.generativeai as genai
import json
import os
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

st.set_page_config(
    page_title="Research Summarizer",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Paper Summarizer")
st.markdown("Upload or paste any research paper "
            "and get an instant AI-powered summary "
            "using Google Gemini.")
st.markdown("---")

# API Key
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = st.sidebar.text_input(
        "Gemini API Key:",
        type="password",
        placeholder="Get free key at "
                    "aistudio.google.com")

HISTORY_FILE = "paper_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_history(data):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(data, f, indent=2)

if 'history' not in st.session_state:
    st.session_state.history = load_history()

def extract_pdf_text(pdf_file):
    if not PDF_AVAILABLE:
        return "PyPDF2 not installed."
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text   = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return "Error extracting PDF: " + str(e)

def summarize_paper(text, summary_type,
                     field, depth, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        'gemini-1.5-flash')

    depth_instructions = {
        "Quick (1-2 min read)":
            "Keep each section to 2-3 bullet "
            "points maximum. Be very concise.",
        "Standard (5 min read)":
            "Provide 4-6 bullet points per "
            "section with moderate detail.",
        "Deep (10 min read)":
            "Be thorough with 6-8 points per "
            "section including technical details."
    }

    type_instructions = {
        "Full Summary":
            "Analyze all sections below.",
        "Key Findings Only":
            "Focus only on key findings "
            "and contributions.",
        "Methods & Results":
            "Focus on methodology, "
            "experiments and results.",
        "Literature Review":
            "Summarize related work and "
            "how it positions this paper."
    }

    prompt = (
        "You are an expert academic research "
        "assistant specializing in " + field +
        ".\n\n"
        "Analyze this research paper and "
        "provide a structured summary.\n"
        "Summary type: " + summary_type + "\n"
        "Instructions: " +
        type_instructions[summary_type] + "\n"
        "Depth: " +
        depth_instructions[depth] + "\n\n"
        "Structure your response as JSON with "
        "these keys:\n"
        "{\n"
        "  \"title\": \"paper title or "
        "'Unknown'\",\n"
        "  \"authors\": \"authors or "
        "'Not specified'\",\n"
        "  \"year\": \"publication year or "
        "'Unknown'\",\n"
        "  \"venue\": \"journal/conference "
        "or 'Unknown'\",\n"
        "  \"one_liner\": \"one sentence "
        "describing the paper\",\n"
        "  \"problem\": [\"bullet points "
        "about the problem addressed\"],\n"
        "  \"methodology\": [\"bullet points "
        "about methods used\"],\n"
        "  \"key_findings\": [\"bullet points "
        "of main results\"],\n"
        "  \"contributions\": [\"novel "
        "contributions to the field\"],\n"
        "  \"limitations\": [\"weaknesses "
        "or limitations\"],\n"
        "  \"future_work\": [\"suggested "
        "future directions\"],\n"
        "  \"keywords\": [\"5-8 keywords\"],\n"
        "  \"difficulty\": \"Easy/Medium/"
        "Hard/Expert\",\n"
        "  \"recommended_for\": \"who should "
        "read this paper\"\n"
        "}\n\n"
        "Return ONLY valid JSON.\n\n"
        "PAPER TEXT (first 6000 chars):\n" +
        text[:6000]
    )

    response = model.generate_content(prompt)
    return response.text

def parse_response(text):
    try:
        clean = text.strip()
        if clean.startswith('```'):
            lines = clean.split('\n')
            clean = '\n'.join(lines[1:-1])
        return json.loads(clean)
    except Exception:
        return None

def generate_qa(text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        'gemini-1.5-flash')
    prompt = (
        "Based on this research paper, "
        "generate 5 insightful questions "
        "and answers that test deep "
        "understanding.\n\n"
        "Format as JSON:\n"
        "{\n"
        "  \"qa\": [\n"
        "    {\"q\": \"question\", "
        "\"a\": \"answer\"}\n"
        "  ]\n"
        "}\n\n"
        "Return ONLY valid JSON.\n\n"
        "PAPER:\n" + text[:4000]
    )
    response = model.generate_content(prompt)
    return response.text

def compare_papers(text1, text2, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        'gemini-1.5-flash')
    prompt = (
        "Compare these two research papers "
        "and provide a structured comparison.\n\n"
        "Format as JSON:\n"
        "{\n"
        "  \"similarities\": [\"...\"],\n"
        "  \"differences\": [\"...\"],\n"
        "  \"paper1_strengths\": [\"...\"],\n"
        "  \"paper2_strengths\": [\"...\"],\n"
        "  \"recommendation\": \"which to read "
        "first and why\"\n"
        "}\n\n"
        "Return ONLY valid JSON.\n\n"
        "PAPER 1:\n" + text1[:2500] +
        "\n\nPAPER 2:\n" + text2[:2500]
    )
    response = model.generate_content(prompt)
    return response.text

# Sidebar
st.sidebar.header("⚙️ Settings")
summary_type = st.sidebar.selectbox(
    "Summary Type:",
    ["Full Summary",
     "Key Findings Only",
     "Methods & Results",
     "Literature Review"])

field = st.sidebar.selectbox(
    "Research Field:",
    ["Machine Learning / AI",
     "Computer Science",
     "Data Science",
     "Natural Language Processing",
     "Computer Vision",
     "Reinforcement Learning",
     "Mathematics / Statistics",
     "Biology / Bioinformatics",
     "Physics", "General / Other"])

depth = st.sidebar.selectbox(
    "Summary Depth:",
    ["Quick (1-2 min read)",
     "Standard (5 min read)",
     "Deep (10 min read)"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Session Stats")
st.sidebar.metric(
    "Papers Summarized",
    len(st.session_state.history))

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Summarize Paper",
    "❓ Q&A Mode",
    "⚖️ Compare Papers",
    "📚 History",
    "💡 Reading Guide"
])

# Tab 1 — Summarize
with tab1:
    st.markdown("### 📄 Summarize a Paper")

    input_method = st.radio(
        "Input Method:",
        ["📋 Paste Text",
         "📂 Upload PDF",
         "🔗 Try Sample Paper"],
        horizontal=True)

    paper_text = ""

    if input_method == "📋 Paste Text":
        paper_text = st.text_area(
            "Paste paper text here:",
            placeholder="Paste abstract, "
                        "introduction or full "
                        "paper text...",
            height=250)

    elif input_method == "📂 Upload PDF":
        if PDF_AVAILABLE:
            uploaded = st.file_uploader(
                "Upload PDF:",
                type=['pdf'])
            if uploaded:
                with st.spinner(
                    "Extracting text from PDF..."
                ):
                    paper_text = extract_pdf_text(
                        uploaded)
                st.success(
                    "Extracted " +
                    str(len(paper_text)) +
                    " characters from PDF")
                with st.expander(
                    "Preview extracted text"
                ):
                    st.text(paper_text[:1000] +
                            "...")
        else:
            st.error(
                "PyPDF2 not installed. "
                "Run: pip install PyPDF2")

    else:  # Sample paper
        sample_papers = {
            "Attention Is All You Need "
            "(Transformer)":
                """Title: Attention Is All You Need
Authors: Vaswani et al.
Year: 2017
Venue: NeurIPS

Abstract:
The dominant sequence transduction models
are based on complex recurrent or
convolutional neural networks that include
an encoder and a decoder. The best
performing models also connect the encoder
and decoder through an attention mechanism.
We propose a new simple network
architecture, the Transformer, based solely
on attention mechanisms, dispensing with
recurrence and convolutions entirely.
Experiments on two machine translation
tasks show these models to be superior in
quality while being more parallelizable
and requiring significantly less time to
train.

Introduction:
Recurrent neural networks, long short-term
memory and gated recurrent neural networks
in particular, have been firmly established
as state of the art approaches in sequence
modeling and transduction problems such as
language modeling and machine translation.
Numerous efforts have since continued to
push the boundaries of recurrent language
models and encoder-decoder architectures.

The Transformer architecture uses
multi-head self-attention to compute
representations of its input and output
without using sequence-aligned RNNs or
convolution. The Transformer achieves 28.4
BLEU on the WMT 2014 English-to-German
translation task, improving over the
existing best results including ensembles
by over 2 BLEU. On the WMT 2014
English-to-French translation task, our
model establishes a new single-model
state-of-the-art BLEU score of 41.0.

Key contributions:
1. Self-attention mechanism replaces RNNs
2. Multi-head attention for parallel computation
3. Positional encoding for sequence order
4. Scaled dot-product attention
5. Superior performance with less training time""",

            "BERT: Pre-training of Deep "
            "Bidirectional Transformers":
                """Title: BERT: Pre-training of Deep
Bidirectional Transformers for Language
Understanding
Authors: Devlin, Chang, Lee, Toutanova
Year: 2019
Venue: NAACL

Abstract:
We introduce a new language representation
model called BERT, which stands for
Bidirectional Encoder Representations from
Transformers. Unlike recent language
representation models, BERT is designed to
pre-train deep bidirectional representations
from unlabeled text by jointly conditioning
on both left and right context in all layers.
As a result, the pre-trained BERT model can
be fine-tuned with just one additional output
layer to create state-of-the-art models for a
wide range of tasks, such as question
answering and language inference, without
substantial task-specific architecture
modifications.

BERT advances the state of the art for eleven
NLP tasks, including pushing the GLUE score to
80.5%, MultiNLI accuracy to 86.7%, SQuAD v1.1
question answering Test F1 to 93.2 and SQuAD
v2.0 Test F1 to 83.1.

Methodology:
BERT uses masked language modeling (MLM) and
next sentence prediction (NSP) as pre-training
objectives. The model is then fine-tuned on
downstream tasks. The architecture is a
multi-layer bidirectional Transformer encoder
based on the original Transformer implementation.""",

            "ResNet: Deep Residual Learning":
                """Title: Deep Residual Learning
for Image Recognition
Authors: He, Zhang, Ren, Sun
Year: 2016
Venue: CVPR

Abstract:
Deeper neural networks are more difficult to
train. We present a residual learning
framework to ease the training of networks
that are substantially deeper than those used
previously. We explicitly reformulate the
layers as learning residual functions with
reference to the layer inputs, instead of
learning unreferenced functions.

Results show that these residual networks are
easier to optimize, and can gain accuracy from
considerably increased depth. On the ImageNet
dataset we evaluate residual nets with a depth
of up to 152 layers - 8x deeper than VGG nets
but still having lower complexity. An ensemble
of these residual nets achieves 3.57% error on
the ImageNet test set. This result won the
1st place on the ILSVRC 2015 classification
task.

Key Innovation:
Skip connections (residual connections) allow
gradients to flow directly through the network,
solving the vanishing gradient problem in very
deep networks. The identity shortcut connections
add neither extra parameter nor computational
complexity."""
        }

        selected_sample = st.selectbox(
            "Choose a sample paper:",
            list(sample_papers.keys()))
        paper_text = sample_papers[
            selected_sample]
        st.text_area(
            "Sample Paper Text:",
            value=paper_text[:500] + "...",
            height=120,
            disabled=True)

    if not api_key:
        st.warning(
            "Enter your Gemini API key "
            "in the sidebar!")
    elif paper_text.strip():
        if st.button(
            "🚀 Summarize Paper",
            type="primary",
            use_container_width=True
        ):
            with st.spinner(
                "🤖 Gemini is reading and "
                "summarizing the paper..."
            ):
                try:
                    raw = summarize_paper(
                        paper_text,
                        summary_type,
                        field, depth,
                        api_key)
                    result = parse_response(raw)

                    if result:
                        st.markdown("---")

                        # Header
                        st.markdown(
                            "<div style='"
                            "background:#162032;"
                            "border:2px solid "
                            "#3498db;"
                            "border-radius:12px;"
                            "padding:20px;"
                            "margin:10px 0'>"
                            "<h2 style='color:"
                            "#58a6ff;margin:0'>" +
                            result.get(
                                'title',
                                'Unknown Title') +
                            "</h2>"
                            "<p style='color:"
                            "#8b949e;margin:6px 0'>"
                            "✍️ " +
                            result.get(
                                'authors',
                                'Unknown') +
                            " | 📅 " +
                            str(result.get(
                                'year',
                                'Unknown')) +
                            " | 📖 " +
                            result.get(
                                'venue',
                                'Unknown') +
                            "</p>"
                            "<p style='color:"
                            "#cdd9e5;"
                            "font-style:italic'>"
                            "" + result.get(
                                'one_liner',
                                '') +
                            "</p>"
                            "</div>",
                            unsafe_allow_html=True)

                        # Metadata
                        c1, c2, c3 = st.columns(3)
                        c1.metric(
                            "Difficulty",
                            result.get(
                                'difficulty',
                                'Unknown'))
                        c2.metric(
                            "Field", field)
                        c3.metric(
                            "Keywords",
                            str(len(result.get(
                                'keywords', []))))

                        st.markdown(
                            "**📖 Recommended "
                            "for:** " +
                            result.get(
                                'recommended_for',
                                'General readers'))

                        # Keywords
                        if result.get('keywords'):
                            kw_str = " ".join([
                                "`" + k + "`"
                                for k in
                                result['keywords']
                            ])
                            st.markdown(
                                "**🏷️ Keywords:** "
                                + kw_str)

                        st.markdown("---")

                        # Sections
                        sections = [
                            ("🎯 Problem Addressed",
                             "problem"),
                            ("🔬 Methodology",
                             "methodology"),
                            ("📊 Key Findings",
                             "key_findings"),
                            ("💡 Contributions",
                             "contributions"),
                            ("⚠️ Limitations",
                             "limitations"),
                            ("🔭 Future Work",
                             "future_work")
                        ]

                        col1, col2 = st.columns(2)
                        for i, (title, key) in \
                                enumerate(sections):
                            with col1 if i % 2 == 0 \
                                    else col2:
                                st.markdown(
                                    "#### " + title)
                                items = result.get(
                                    key, [])
                                if isinstance(
                                    items, list
                                ):
                                    for item in items:
                                        st.markdown(
                                            "• " +
                                            str(item))
                                else:
                                    st.markdown(
                                        str(items))

                        # Save to history
                        st.session_state\
                            .history.append({
                            'timestamp': str(
                                datetime.now()
                            )[:16],
                            'title': result.get(
                                'title',
                                'Unknown'),
                            'authors': result.get(
                                'authors', ''),
                            'year': result.get(
                                'year', ''),
                            'field': field,
                            'one_liner':
                                result.get(
                                    'one_liner',
                                    ''),
                            'keywords':
                                result.get(
                                    'keywords', []),
                            'difficulty':
                                result.get(
                                    'difficulty',
                                    ''),
                            'full_result': result
                        })
                        save_history(
                            st.session_state
                            .history)

                        # Download
                        summary_text = (
                            "PAPER: " +
                            result.get(
                                'title', '') +
                            "\n"
                            "AUTHORS: " +
                            result.get(
                                'authors', '') +
                            "\n"
                            "YEAR: " +
                            str(result.get(
                                'year', '')) +
                            "\n\n"
                            "ONE LINER: " +
                            result.get(
                                'one_liner',
                                '') + "\n\n"
                        )
                        for title, key in sections:
                            summary_text += (
                                title + ":\n")
                            items = result.get(
                                key, [])
                            if isinstance(
                                items, list
                            ):
                                for item in items:
                                    summary_text += \
                                        "  - " + \
                                        str(item) + \
                                        "\n"
                            summary_text += "\n"

                        st.download_button(
                            "⬇️ Download Summary",
                            summary_text,
                            "paper_summary.txt",
                            "text/plain")
                        st.success(
                            "✅ Saved to history!")
                    else:
                        st.error(
                            "Could not parse "
                            "response. Try again.")
                except Exception as e:
                    st.error("Error: " + str(e))
    else:
        st.info(
            "Paste paper text or upload a PDF "
            "to get started.")

# Tab 2 — Q&A
with tab2:
    st.markdown("### ❓ Paper Q&A Mode")
    st.markdown(
        "Generate comprehension questions "
        "to test your understanding.")

    qa_text = st.text_area(
        "Paste paper text for Q&A:",
        placeholder="Paste the paper text here...",
        height=200,
        key="qa_text")

    if not api_key:
        st.warning("Enter Gemini API key!")
    elif qa_text.strip():
        if st.button(
            "❓ Generate Q&A",
            type="primary"
        ):
            with st.spinner(
                "Generating questions..."
            ):
                try:
                    raw = generate_qa(
                        qa_text, api_key)
                    result = parse_response(raw)

                    if result and \
                            result.get('qa'):
                        st.markdown(
                            "### 📝 Questions "
                            "& Answers")
                        for i, qa in enumerate(
                            result['qa'], 1
                        ):
                            with st.expander(
                                "Q" + str(i) +
                                ": " +
                                qa.get('q', '')
                            ):
                                st.markdown(
                                    "**Answer:** " +
                                    qa.get('a', ''))
                    else:
                        st.error(
                            "Could not generate "
                            "Q&A. Try again.")
                except Exception as e:
                    st.error("Error: " + str(e))

# Tab 3 — Compare
with tab3:
    st.markdown("### ⚖️ Compare Two Papers")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📄 Paper 1")
        text1 = st.text_area(
            "Paste Paper 1:",
            height=200,
            key="paper1",
            placeholder="Paste first paper...")
    with col2:
        st.markdown("#### 📄 Paper 2")
        text2 = st.text_area(
            "Paste Paper 2:",
            height=200,
            key="paper2",
            placeholder="Paste second paper...")

    if not api_key:
        st.warning("Enter Gemini API key!")
    elif text1.strip() and text2.strip():
        if st.button(
            "⚖️ Compare Papers",
            type="primary",
            use_container_width=True
        ):
            with st.spinner(
                "Comparing papers..."
            ):
                try:
                    raw = compare_papers(
                        text1, text2, api_key)
                    result = parse_response(raw)

                    if result:
                        st.markdown("---")

                        col_a, col_b = \
                            st.columns(2)

                        with col_a:
                            st.markdown(
                                "#### ✅ "
                                "Similarities")
                            for s in result.get(
                                'similarities', []
                            ):
                                st.markdown(
                                    "• " + s)

                            st.markdown(
                                "#### 💪 "
                                "Paper 1 Strengths")
                            for s in result.get(
                                'paper1_strengths',
                                []
                            ):
                                st.markdown(
                                    "• " + s)

                        with col_b:
                            st.markdown(
                                "#### 🔀 "
                                "Differences")
                            for d in result.get(
                                'differences', []
                            ):
                                st.markdown(
                                    "• " + d)

                            st.markdown(
                                "#### 💪 "
                                "Paper 2 Strengths")
                            for s in result.get(
                                'paper2_strengths',
                                []
                            ):
                                st.markdown(
                                    "• " + s)

                        st.markdown("---")
                        st.info(
                            "📖 **Recommendation:** "
                            + result.get(
                                'recommendation',
                                ''))
                    else:
                        st.error(
                            "Could not parse "
                            "comparison.")
                except Exception as e:
                    st.error("Error: " + str(e))

# Tab 4 — History
with tab4:
    st.markdown("### 📚 Paper History")

    if not st.session_state.history:
        st.info(
            "No papers summarized yet!")
    else:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            save_history([])
            st.rerun()

        st.markdown(
            "**" +
            str(len(st.session_state.history)) +
            " papers summarized**")

        for entry in reversed(
            st.session_state.history
        ):
            with st.expander(
                "📄 " +
                entry.get('title', 'Unknown') +
                " (" +
                str(entry.get('year', '')) +
                ") — " +
                entry.get('timestamp', '')
            ):
                st.markdown(
                    "**Authors:** " +
                    entry.get('authors', ''))
                st.markdown(
                    "**Field:** " +
                    entry.get('field', ''))
                st.markdown(
                    "**Difficulty:** " +
                    entry.get('difficulty', ''))
                st.markdown(
                    "_" +
                    entry.get(
                        'one_liner', '') + "_")

                if entry.get('keywords'):
                    kw = " ".join([
                        "`" + k + "`"
                        for k in
                        entry['keywords']
                    ])
                    st.markdown(
                        "**Keywords:** " + kw)

                if entry.get('full_result'):
                    findings = entry[
                        'full_result'].get(
                        'key_findings', [])
                    if findings:
                        st.markdown(
                            "**Key Findings:**")
                        for f in findings[:3]:
                            st.markdown(
                                "• " + str(f))

        # Summary stats
        st.markdown("---")
        st.markdown("#### 📊 Reading Stats")
        fields = [
            e.get('field', 'Unknown')
            for e in st.session_state.history
        ]
        if fields:
            import pandas as pd
            import plotly.express as px
            field_counts = pd.Series(
                fields).value_counts()
            fig = px.pie(
                values=field_counts.values,
                names=field_counts.index,
                title='Papers by Field')
            fig.update_layout(height=300)
            st.plotly_chart(
                fig,
                use_container_width=True)

# Tab 5 — Reading Guide
with tab5:
    st.markdown("### 💡 How to Read Research Papers")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 📖 The Three-Pass Method

        **First Pass (5-10 min)**
        Read title, abstract, intro,
        section headers and conclusion.
        Answer: What problem? What approach?
        What result?

        **Second Pass (1 hour)**
        Read carefully, skipping proofs.
        Note key figures and tables.
        Mark what you don't understand.

        **Third Pass (4-5 hours)**
        Understand every detail.
        Virtually re-implement the paper.
        Identify assumptions and weaknesses.

        #### 🎯 Where to Find Papers
        - **arXiv.org** — Free preprints
        - **Semantic Scholar** — AI-powered search
        - **Google Scholar** — Citation tracking
        - **Papers With Code** — Code + papers
        - **HuggingFace** — NLP/CV papers
        """)

    with col2:
        st.markdown("""
        #### 📊 Key Sections to Focus On

        **Abstract**
        Problem, approach, result in 1 paragraph.
        Tells you if worth reading.

        **Introduction**
        Motivation, related work gap,
        contributions and paper structure.

        **Related Work**
        Prior art. Shows how this paper
        differs from existing work.

        **Methodology**
        The core technical contribution.
        Read this most carefully.

        **Experiments**
        Datasets, baselines, ablations,
        metrics. Are results convincing?

        **Conclusion**
        Summary and future directions.
        Often repeats abstract.

        #### ⚡ Critical Reading Questions
        - What problem does it solve?
        - Is the evaluation fair?
        - What are the assumptions?
        - Can results be reproduced?
        - What is not shown?
        """)

    st.markdown("---")
    st.markdown("#### 🏆 Must-Read Papers for ML Students")
    must_read = [
        ("Attention Is All You Need",
         "Vaswani et al. 2017",
         "Transformer architecture"),
        ("BERT", "Devlin et al. 2019",
         "Pre-trained language models"),
        ("ResNet", "He et al. 2016",
         "Residual connections in CNNs"),
        ("GAN", "Goodfellow et al. 2014",
         "Generative adversarial networks"),
        ("Dropout", "Srivastava et al. 2014",
         "Regularization technique"),
        ("Adam Optimizer",
         "Kingma & Ba 2015",
         "Adaptive learning rates"),
        ("Word2Vec", "Mikolov et al. 2013",
         "Word embeddings"),
        ("AlphaFold", "Jumper et al. 2021",
         "Protein structure prediction"),
    ]

    import pandas as pd
    must_df = pd.DataFrame(
        must_read,
        columns=['Paper', 'Citation', 'Key Idea'])
    st.dataframe(
        must_df,
        use_container_width=True,
        hide_index=True)

st.markdown("---")
st.markdown(
    "Built by **Jyotiraditya** | "
    "Research Paper Summarizer | "
    "Powered by Google Gemini"
)