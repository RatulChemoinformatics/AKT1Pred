import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
from streamlit_option_menu import option_menu

# Add another image logo
another_image = Image.open('AMRITA.png')


# Set page configuration
st.set_page_config(page_title='QSAR ToolBox', page_icon='🌐', layout="wide")

# Create title and subtitle
html_temp = """
        <div style="background-color:teal">
        <h1 style="font-family:arial;color:white;text-align:center;">AKT1-Pred</h1>
        <h4 style="font-family:arial;color:white;text-align:center;">Artificial Intelligence Based Bioactivity Prediction Web-App</h4>       
        </div>
        <br>
        """
st.markdown(html_temp, unsafe_allow_html=True)

# Logo image
image = Image.open('AKT1Pred.png')
st.image(image, use_column_width=True)

# Page title
st.markdown("""
# Application Details

This app allows you to predict the bioactivity of any molecule towards inhibiting AKT1.

**Credits**
- App built in `Python` + `Streamlit` by [Ratul Bhowmik, Dr. C. Gopi Mohan, Sunil Kumar, Dr. Bijo Mathew]
- Author affiliations [Bioinformatics and Computational Biology Lab, Amrita School of Nanosciences and Molecular Medicine, Amrita Vishwa Vidyapeetham, Kochi]
- Descriptor calculated using [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) [[Read the Paper]](https://doi.org/10.1002/jcc.21707).

- **To be a registered user (free of charge) of this site for academic/commercial purposes, kindly download from here (http://surl.li/kcyed) and sign a one-time License Agreement Form and send it to cgmohan@acnsmm.amrita.edu
---
""")

# loading the saved models
bioactivity_first_model = pickle.load(open('akt1_pubchem.pkl', 'rb'))
bioactivity_second_model = pickle.load(open('akt1_substructure.pkl', 'rb'))


# For hiding streamlit messages
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)

st.sidebar.title("AKT1-Pred")
st.sidebar.header("Menu")
CB = st.sidebar.checkbox("Web Application Information")

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Artificial Intelligence assisted Machine Learning Bioactivity Prediction Models',
                           ['AKT1 prediction model using pubchemfingerprints',
                            'AKT1 prediction model using substructurefingerprints'],
                           icons=['activity', 'activity'],
                           default_index=0)

# AKT1 prediction model using pubchemfingerprints
if selected == 'AKT1 prediction model using pubchemfingerprints':
    # page title
    st.title('Predict bioactivity of molecules against AKT1 using pubchemfingerprints')

    # Molecular descriptor calculator
    def desc_calc():
        # Performs the descriptor calculation
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove('molecule.smi')

    # File download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
        return href

    # Model building
    def build_model(input_data):
        # Apply model to make predictions
        prediction = bioactivity_first_model.predict(input_data)
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pIC50')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

    # Sidebar
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
        st.sidebar.markdown("""
        [Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
        """)

    if st.sidebar.button('Predict'):
        if uploaded_file is not None:
            load_data = pd.read_table(uploaded_file, sep=' ', header=None)
            load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

            st.header('**Original input data**')
            st.write(load_data)

            with st.spinner("Calculating descriptors..."):
                desc_calc()

            # Read in calculated descriptors and display the dataframe
            st.header('**Calculated molecular descriptors**')
            desc = pd.read_csv('descriptors_output.csv')
            st.write(desc)
            st.write(desc.shape)

            # Read descriptor list used in previously built model
            st.header('**Subset of descriptors from previously built models**')
            Xlist = list(pd.read_csv('akt1_pubchem.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            st.write(desc_subset.shape)

            # Apply trained model to make prediction on query compounds
            build_model(desc_subset)
        else:
            st.warning('Please upload an input file.')

# AKT1 prediction model using substructurefingerprints
elif selected == 'AKT1 prediction model using substructurefingerprints':
    # page title
    st.title('Predict bioactivity of molecules against AKT1 using substructurefingerprints')

    # Molecular descriptor calculator
    def desc_calc():
        # Performs the descriptor calculation
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/SubstructureFingerprinter.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove('molecule.smi')

    # File download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
        return href

    # Model building
    def build_model(input_data):
        # Apply model to make predictions
        prediction = bioactivity_second_model.predict(input_data)
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pIC50')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

    # Sidebar
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
        st.sidebar.markdown("""
        [Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
        """)

    if st.sidebar.button('Predict'):
        if uploaded_file is not None:
            load_data = pd.read_table(uploaded_file, sep=' ', header=None)
            load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

            st.header('**Original input data**')
            st.write(load_data)

            with st.spinner("Calculating descriptors..."):
                desc_calc()

            # Read in calculated descriptors and display the dataframe
            st.header('**Calculated molecular descriptors**')
            desc = pd.read_csv('descriptors_output.csv')
            st.write(desc)
            st.write(desc.shape)

            # Read descriptor list used in previously built model
            st.header('**Subset of descriptors from previously built models**')
            Xlist = list(pd.read_csv('akt1_substructure.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            st.write(desc_subset.shape)

            # Apply trained model to make prediction on query compounds
            build_model(desc_subset)
        else:
            st.warning('Please upload an input file.')
