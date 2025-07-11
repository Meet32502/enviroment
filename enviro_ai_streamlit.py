# enviro_ai_streamlit.py

# 1. Imports
import streamlit as st
from transformers import pipeline
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# 2. Load models
st.info("🔄 Loading AI models... This might take a moment.")

@st.cache_resource
def load_models():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
    fill_mask = pipeline("fill-mask", model="roberta-base")

    if torch.cuda.is_available():
        image_gen = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
        ).to("cuda")
    else:
        image_gen = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        ).to("cpu")
    return classifier, ner, fill_mask, image_gen

classifier, ner, fill_mask, image_gen = load_models()

st.success("✅ All models loaded successfully!")

# 3. Define functions (mostly copied, with minor adjustments for Streamlit compatibility)
def classify_text(text):
    labels = ["Waste Management", "Water Management", "Air Pollution", "Recycling", "Energy Conservation"]
    result = classifier(text, candidate_labels=labels)
    return f"Category: **{result['labels'][0]}**, Score: **{result['scores'][0]:.2f}**"

def generate_image(prompt):
    image = image_gen(prompt).images[0]  # returns PIL.Image
    return image

ENV_TERMS = [
    # Pollution
    "pollution", "air", "water", "soil", "noise", "radiation", "smog", "contamination",
    "runoff", "eutrophication", "emission", "discharge",
    
    # Waste
    "waste", "garbage", "litter", "plastic", "microplastic", "sewage", "industrial",
    "landfill", "toxic", "hazardous", "e-waste", "compost", "recyclables",

    # Natural Resources
    "river", "ocean", "lake", "forest", "biodiversity", "wildlife", "wetland",
    "coral", "marine", "vegetation", "habitat",

    # Climate
    "climate", "warming", "greenhouse", "carbon", "dioxide", "methane", "ozone",
    "temperature", "drought", "flood", "acid rain", "co2", "ghg",

    # Sustainability
    "recycle", "reuse", "compost", "conservation", "renewable", "solar", "wind",
    "hydro", "clean energy", "green tech", "sustainable", "biodegradable",
    "energy efficiency", "net zero",

    # Industrial
    "factory", "industry", "agriculture", "vehicle", "transport", "plant",
    "chemical", "refinery", "pesticide", "fertilizer", "manufacturing"
]

def ner_with_graph(text):
    entities = ner(text)
    G = nx.Graph()

    seen_words = set()

    # Add NER entities
    for ent in entities:
        raw_label = ent["entity_group"]
        clean_label = {
            "PER": "Person", "LOC": "Location", "ORG": "Organization", "MISC": "Misc"
        }.get(raw_label, raw_label)

        word = ent["word"].replace("##", "").strip()
        if word not in seen_words:
            G.add_node(word, label=clean_label)
            seen_words.add(word)

    # Add environmental terms
    words = text.split()
    for word in words:
        clean_word = word.lower().strip(",.")
        if clean_word in ENV_TERMS:
            if word not in seen_words:
                G.add_node(clean_word, label="Environmental") # Use clean_word for node if it's an ENV_TERM
                seen_words.add(clean_word)

    # Link nodes in order of appearance (this logic might need refinement for more complex graphs,
    # but for sequential entities it's okay)
    # Let's adjust to link entities that appear close to each other
    text_words = [w.lower().strip(",.") for w in text.split()]
    for i in range(len(text_words)):
        for j in range(i + 1, min(i + 4, len(text_words))): # Link words within a small window
            word1 = text_words[i]
            word2 = text_words[j]
            if word1 in seen_words and word2 in seen_words and word1 != word2:
                G.add_edge(word1, word2)

    # Draw the graph
    fig, ax = plt.subplots(figsize=(10, 7)) # Increased figure size for better visibility
    
    # Use a more visually appealing layout
    try:
        pos = nx.spring_layout(G, k=0.7, iterations=50, seed=42) # k adjusts optimal distance
    except Exception: # Fallback for graphs with very few nodes
        pos = nx.circular_layout(G)


    labels = nx.get_node_attributes(G, "label")
    color_map = {
        "Person": "#FF7F50",  # Coral
        "Location": "#6495ED",  # CornflowerBlue
        "Organization": "#8FBC8F",  # DarkSeaGreen
        "Misc": "#BA55D3",  # MediumOrchid
        "Environmental": "#20B2AA"  # LightSeaGreen
    }
    node_colors = [color_map.get(labels.get(node, ""), "#D3D3D3") for node in G.nodes] # LightGray for unknown

    nx.draw(
        G, pos, with_labels=False, node_color=node_colors, edge_color="gray",
        font_size=9, node_size=3000, font_weight="bold", width=1.5,
        alpha=0.8 # Slightly transparent nodes
    )
    # Draw labels separately for better control
    nx.draw_networkx_labels(G, pos, labels={n: f"{n}\n({labels[n]})" for n in labels}, font_size=8)


    # Create a legend
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                 markerfacecolor=color, markersize=10)
                      for label, color in color_map.items()]
    ax.legend(handles=legend_handles, title="Entity Types", loc="upper left", bbox_to_anchor=(1, 1))
    
    plt.title("🌳 Environmental Entity Relationship Graph", size=14)
    plt.axis('off') # Hide axes

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def fill_blank(text):
    if "<mask>" not in text:
        st.error("❌ Please include exactly one `<mask>` token in your sentence.")
        return []
    try:
        results = fill_mask(text)
        return [r["sequence"] for r in results]
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return []

# 4. Streamlit UI
st.set_page_config(
    page_title="EcoInsight AI",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #006400; /* Dark Green */
        text-align: center;
    }
    h2 {
        color: #2E8B57; /* Sea Green */
    }
    .stButton>button {
        background-color: #28A745; /* Green */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #218838; /* Darker Green on hover */
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        border: 1px solid #ced4da;
        padding: 10px;
    }
    .stTextArea>div>div>textarea {
        border-radius: 5px;
        border: 1px solid #ced4da;
        padding: 10px;
    }
    .css-1d391kg { /* Target sidebar for better styling if needed */
        background-color: #e0ffe0; /* Light green sidebar */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌱 EcoInsight AI: Smart Environmental Assistant")

st.markdown(
    """
    Welcome to EcoInsight AI! This intelligent dashboard helps you analyze environmental text,
    generate relevant images, visualize entity relationships, and complete sentences.
    Explore the tabs below to utilize different AI capabilities.
    """
)

# Using st.tabs for a cleaner layout
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Eco-Sentence Classification",
    "🖼️ Eco-Image Weaver",
    "📊 Entity Eco-Grapher",
    "🧩 Eco-Predictive Text"
])

with tab1:
    st.header("📝 Eco-Sentence Classification")
    st.markdown("Enter a sentence related to environmental topics to categorize it.")
    inp = st.text_area("Environmental Statement", placeholder="e.g., The city implemented new policies for waste segregation.", height=100)
    if st.button("🚀 Analyze Statement"):
        if inp:
            with st.spinner("Classifying..."):
                result = classify_text(inp)
                st.markdown(f"**Classification Result:** {result}")
        else:
            st.warning("Please enter some text to classify.")

with tab2:
    st.header("🖼️ Eco-Image Weaver")
    st.markdown("Describe an environmental scene, and AI will generate an image for you.")
    img_prompt = st.text_area("Image Description", placeholder="e.g., A lush green forest with a clear river and diverse wildlife.", height=100)
    if st.button("🧠 Weave Image"):
        if img_prompt:
            with st.spinner("Generating image... This might take a few moments."):
                image = generate_image(img_prompt)
                st.image(image, caption="Generated Eco-Image", use_column_width=True)
        else:
            st.warning("Please enter a description for image generation.")

with tab3:
    st.header("📊 Entity Eco-Grapher")
    st.markdown("Input a sentence to extract and visualize named entities and environmental terms as a graph.")
    ner_input = st.text_area("Sentence for Entity Analysis", placeholder="e.g., Dr. Smith at the EPA discussed plastic pollution in the Atlantic Ocean.", height=100)
    if st.button("📈 Visualize Entities"):
        if ner_input:
            with st.spinner("Drawing entity graph..."):
                graph_image = ner_with_graph(ner_input)
                st.image(graph_image, caption="Eco-Entity Relationship Map", use_column_width=True)
        else:
            st.warning("Please enter a sentence for entity extraction.")

with tab4:
    st.header("🧩 Eco-Predictive Text")
    st.markdown("Fill in the blank (use `<mask>`) in an environmental sentence.")
    fill_input = st.text_input("Sentence with a missing word (use <mask>)", value="Excessive <mask> is harmful to marine life.")
    if st.button("🔍 Predict Missing Word"):
        if fill_input:
            with st.spinner("Predicting..."):
                predictions = fill_blank(fill_input)
                if predictions:
                    st.markdown("**Top Predictions:**")
                    for i, pred in enumerate(predictions):
                        st.write(f"- {pred}")
        else:
            st.warning("Please enter a sentence with a `<mask>` token.")

st.sidebar.header("About EcoInsight AI")
st.sidebar.markdown(
    "This application leverages state-of-the-art AI models to provide insights into environmental data.\n\n"
    "**Models Used:**\n"
    "- **Zero-shot Classification:** `facebook/bart-large-mnli`\n"
    "- **Named Entity Recognition (NER):** `dslim/bert-base-NER`\n"
    "- **Fill-Mask:** `roberta-base`\n"
    "- **Image Generation:** `runwayml/stable-diffusion-v1-5`\n\n"
    "Developed with Streamlit and 🤗 Transformers."
)
st.sidebar.image("https://www.flaticon.com/svg/static/icons/svg/2972/2972044.svg", width=100, caption="Eco-friendly Icon")