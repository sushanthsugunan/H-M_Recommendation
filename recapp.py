import streamlit as st
import pandas as pd
import pickle
import surprise
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from pathlib import Path
from typing import Dict, Text

tab1, tab2, tab3, tab4 = st.tabs(["Segmentation", "Model Comparison","CF Recommendation","Two-Tower"])
path = "C:/Users/Sushanth S/Class Lectures/Marketing/HM-Recommender-System-App-main/Project/Models"
# Load it back; can also be done in TensorFlow Serving.
loaded = tf.saved_model.load(path)

with tab1:
    st.header("Market Segmentation")
    Segments = pd.read_csv('Dataset/Segments.csv')
    Segments.drop(['Unnamed: 0'],axis=1,inplace=True)
    #Segments = Segments.reset_index()
    st.text("Market Segements Identified")
    st.table(Segments)
    
    Tree = Image.open('plots/RFM_categories_treemap.png')
    RFM = Image.open('plots/RFM_categories.png')
    Hist = Image.open('plots/RFM_histogram.png')
    Articles = Image.open('plots/articles_desc.png')
    st.image(Tree)
    st.image(RFM,width=500)
    st.image(Hist)
    st.image(Articles)
    

with tab2:
    st.header("Machine Learning Models")
    TwoTower = pd.read_csv('Dataset/Twotower_perf.csv')
    TwoTower = TwoTower.reset_index()
    st.text("Two tower model performance parameters")
    st.table(TwoTower)
    model = Image.open('plots/Model_comparison.png')
    st.text("RMSE Values of 3 models used for collaborative filtering")
    st.image(model)


with tab3:
    # Header
    st.header('CF and Article based Recommendations')
    header_image = Image.open('Images/H&Mbanner.jpg')
    st.image(header_image, width=700)
    sidebar = Image.open('Images/SPJ.png')
    st.sidebar.image(sidebar, width=200)
    st.sidebar.caption('By [Sushanth S | Vinit | Aditya]')

    # Load in appropriate DataFrames, user ratings
    articles_df = pd.read_csv('Dataset/articles.csv', index_col='article_id')
    articles_df2 = pd.read_csv('Dataset/articles.csv')

    # Customer data for collabortive filtering
    df_customer = pd.read_csv('Dataset/df_customer.csv', index_col='customer_id')

    # Meta data for collabortive filtering
    transactions = pd.read_csv('Dataset/out.zip')

    # Meta data for content based
    data = pd.read_csv('Dataset/out_content.zip')
    # Import final collab model
    collab_model = pickle.load(open('Models/collaborative_model.sav', 'rb'))

    def article_recommendation(article_input, n1):
        article = articles_df2[articles_df2['article_id'] == article_input].index
        y = np.array(data.loc[article]).reshape(1, -1)
        cos_sim = cosine_similarity(data, y)
        cos_sim = pd.DataFrame(data=cos_sim, index=data.index)
        cos_sim.sort_values(by = 0, ascending = False, inplace=True)
        results = cos_sim.index.values
        results_df = articles_df2.loc[results]
        results_df.reset_index(inplace=True)
        results_df.rename(columns={'prod_name':'Product Name','product_type_name':'Product Type Name', 'product_group_name':'Product Group Name',
                                'index_group_name':'Index Group Name', 'garment_group_name':'Garment Group Name'}, inplace=True)
        results_df = results_df.iloc[:100, :]
        results_df = results_df[['article_id', 'Product Name', 'Product Type Name', 'Product Group Name', 'Index Group Name', 'Garment Group Name']]
        results_df = results_df.sample(frac=1).reset_index(drop=True)
        return results_df.head(n1)

    def customer_article_recommendation(user,n2):
        have_bought = [pd.to_numeric(df_customer.loc[user, 'article_id'])]
        not_bought = articles_df.copy()
        not_bought.drop(have_bought, inplace=True)
        not_bought.reset_index(inplace=True)
        not_bought['est_purchase'] = not_bought['article_id'].apply(lambda x: collab_model.predict(user, x).est)
        not_bought.sort_values(by='est_purchase', ascending=False, inplace=True)
        
        not_bought.rename(columns={'prod_name':'Product Name','product_type_name':'Product Type Name', 'product_group_name':'Product Group Name',
                                'index_group_name':'Index Group Name', 'garment_group_name':'Garment Group Name'}, inplace=True)
        results_df = not_bought[['article_id','Product Name', 'Product Type Name', 'Product Group Name', 'Index Group Name', 'Garment Group Name']]
        results_df = results_df.sample(frac=1).reset_index(drop=True)
        return results_df.head(n2)
    

    # print the image of the articles recommended by CB
    def print_image_cf(results_cf, n2):
        f, ax = plt.subplots(1, n2, figsize=(100,50))
        i = 0
        article_id_cf = results_cf['article_id']
        for index, data in enumerate(article_id_cf):
            desc = articles_df2[articles_df2['article_id'] == data]['detail_desc'].iloc[0]
            desc_list = desc.split(' ')
            for j, elem in enumerate(desc_list):
                if j > 0 and j % 5 == 0:
                    desc_list[j] = desc_list[j] + '\n'
            desc = ' '.join(desc_list)
            img = mpimg.imread(f'C:/Users/Sushanth S/Downloads/images/0{str(data)[:2]}/0{int(data)}.jpg')
            ax[i].imshow(img)
            ax[i].set_xticks([], [])
            ax[i].set_yticks([], [])
            ax[i].grid(False)
            ax[i].set_xlabel(desc, fontsize=90)
            i += 1
        return plt.show()

    st.sidebar.subheader('We use similiarity based recommendation and collaborative filtering based recommendation')

    st.title('Porduct Recommendation System')
    st.subheader('Based on input customer ID or Article specific products will be recommended')
    st.subheader("See the sidebar navigation for options")

    page_names = ['Customer Based Recommendation', 'Article based Recommendation']
    page = st.sidebar.radio('Navigation', page_names)

    st.sidebar.caption('Please refer codes for more details')
    if page == 'Customer Based Recommendation':
        st.header("User Selected customerID Option")
        
        user = st.text_input("Enter unique Customer ID.")
        n2 = st.number_input("Enter the number of article recommendations you would like.", max_value=20)
        rec_button = st.button("Get Recommendation")
    
        if rec_button:
            results = customer_article_recommendation(user, n2)
            st.table(results)
            result_image = print_image_cf(results, n2)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(result_image)


    else:
        st.header("User selected Similar Articles Option.")
        article_input = st.number_input("Please enter a article ID.", max_value=959461001)
        article = articles_df2.index[articles_df2['article_id'] == article_input]
        n1 = st.number_input("Please enter the number of recommendations you would like.", max_value=20, key=2)
        book_button = st.button("Get some recommendations...", key=3)
        if book_button:
            results2 = article_recommendation(article_input, n1)
            st.table(results2)
            result_image2 = print_image_cf(results2, n1)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(result_image2)
    


with tab4:
    # Header
    st.header('Two Tower Model Recommendations')
    def Twotowerrecommendation(user,n_recs):
       
        df = pd.read_csv("Dataset/combine_customer_matrixr.csv")
        df.drop(['Unnamed: 0'],axis=1,inplace=True)
        df = df.reset_index()
        df['article_id'] =df['article_id'].astype("str")
        df['article_id'] =df['article_id'].apply(lambda x: x.zfill(10))
        # Pass a user id in, get top predicted movie titles back.
        _, titles = loaded([user])
        #titles = titles[0][:n_recs]
        #print(titles)   
        
        df1 = pd.DataFrame(titles.numpy()).astype("int").T
        #print(df1)
        df1.rename(columns = {0:'article_id'}, inplace = True)
        df1['article_id'] =df1['article_id'].astype("str")
        df1['article_id'] =df1['article_id'].apply(lambda x: x.zfill(10))
        lst = df1.article_id.values.tolist()
        #print(lst)
        A = set(df.loc[df['customer_id'] == user].index)
        #print(A)
        B = set(df.loc[df['article_id'].isin(lst)].index)
        #print(len(B))
        C = B - A
        D = df.loc[C].drop_duplicates('article_id').set_index('article_id').drop(["customer_id"],axis=1).reset_index()
        #print(type(D))
        
        return D.head(n_recs)

    
    
    user = st.text_input("Enter Customer ID.")
    n_recs = st.number_input("Enter # of article recommendations you would like.", max_value=10)
    rec_button = st.button("Get Recco.")

    if rec_button:
        results = Twotowerrecommendation(user,n_recs)
        results = results.astype({'article_id':'int'})
        st.table(results)
        result_image = print_image_cf(results, n_recs)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(result_image)