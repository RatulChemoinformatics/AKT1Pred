# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:33:56 2023

@author: RATUL BHOWMIK
"""

import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
from streamlit_option_menu import option_menu

# The App
st.title('💊 Akt1-pred app')
st.info('Akt1-pred allows users to predict bioactivity of a query molecule against the Akt1 target protein.')



# loading the saved models
bioactivity_first_model = pickle.load(open('akt1_pubchem.pkl', 'rb'))
bioactivity_second_model = pickle.load(open('akt1_substructure.pkl', 'rb'))

# Define the tabs
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs(['Main', 'About', 'What is AKT1?', 'Dataset', 'Model performance', 'Python libraries', 'Citing us', 'Application Developers'])

with tab1:
    st.title('Application Description')
    st.success(
        " This module of [**Akt1-pred**](https://github.com/RatulChemoinformatics/AKT1Pred) has been built to predict bioactivity and identify potent inhibitors against Akt1 using robust machine learning algorithms."
    )

# Define a sidebar for navigation
with st.sidebar:
    selected = st.selectbox(
        'Choose a prediction model',
        [
            'AKT1 prediction model using pubchemfingerprints',
            'AKT1 prediction model using substructurefingerprints',
        ],
    )

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
        [Example input file](https://github.com/RatulChemoinformatics/QSAR/blob/main/predict.txt)
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
            
            
            
            
with tab2:
  coverimage = Image.open('AKT1Pred.png')
  st.image(coverimage)
with tab3:
  st.header('What is AKT1?')
  st.write('This gene encodes one of the three members of the human AKT serine-threonine protein kinase family which are often referred to as protein kinase B alpha, beta, and gamma. These highly similar AKT proteins all have an N-terminal pleckstrin homology domain, a serine/threonine-specific kinase domain and a C-terminal regulatory domain. These proteins are phosphorylated by phosphoinositide 3-kinase (PI3K). AKT/PI3K forms a key component of many signalling pathways that involve the binding of membrane-bound ligands such as receptor tyrosine kinases, G-protein coupled receptors, and integrin-linked kinase. These AKT proteins therefore regulate a wide variety of cellular functions including cell proliferation, survival, metabolism, and angiogenesis in both normal and malignant cells.')
with tab4:
  st.header('Dataset')
  st.write('''
    In our work, we retrieved a human Akt1 biological dataset from the ChEMBL database. The data was curated and resulted in a non-redundant set of 3393 Akt1 inhibitors, which demostrated a bioactivity value (pIC50) between 10 to 3.3
    ''')
with tab5:
  st.header('Model performance')
  st.write('We selected a total of 2 different molecular signatures namely pubchem fingerprints and substructure fingerprints to build the web application. The correlation coefficient, RMSE, and MAE values for the pubchem fingerprint model was found to be 0.982, 0.2779, and 0.2165. The correlation coefficient, RMSE, and MAE values for the substructure fingerprint model was found to be 0.9649, 0.3507, and 0.2842.')
with tab6:
  st.header('Python libraries')
  st.markdown('''
    This app is based on the following Python libraries:
    - `streamlit`
    - `pandas`
    - `rdkit`
    - `padelpy`
  ''')
with tab7:
  st.markdown('Kuttappan S, Bhowmik R, Gopi Mohan C. Probing the origins of programmed death ligand-1 inhibition by implementing machine learning-assisted sequential virtual screening techniques, ***Molecular Diversity*** (2023) DOI: https://doi.org/10.1007/s11030-023-10697-5.')
with tab8:
  st.markdown('Ratul Bhowmik, Ajay Manaithiya, Ranajit Nath, Sameer Sharma. [***Department of Bioinformatics, BioNome Private Limited, Bengaluru, Karnataka, India***] ')