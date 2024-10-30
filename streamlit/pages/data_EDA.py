import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cv2
from utils.streamlit_data_loader import load_and_process_data

    
# 데이터 로드
@st.cache_data  # 캐싱을 통한 성능 최적화
def load_data(json_path):
    return load_and_process_data(json_path)

def main():
    st.title('이미지 데이터 분석 대시보드')
        
    # 데이터 로드
    json_path = "../data/chinese_receipt/ufo/train.json"  # train.json 경로 지정
    data, df_images, df_words = load_data(json_path)
    
       
    # 1. 기본 통계량 표시
    st.header('📊 기본 통계')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("총 이미지 수", len(df_images))
    with col2:
        st.metric("평균 단어 수", round(df_images['word_counts'].mean(), 2))
    with col3:
        st.metric("최대 단어 수", df_images['word_counts'].max())

    # 2. 이미지 크기 분포 시각화
    st.header('📐 이미지 크기 분포')
    fig = px.scatter(df_images, 
                    x='image_width', 
                    y='image_height',
                    title='이미지 크기 분포')
    st.plotly_chart(fig)

    # 3. 단어 수 분포 히스토그램
    st.header('📝 단어 수 분포')
    fig_hist = px.histogram(df_images, 
                           x='word_counts',
                           title='이미지당 단어 수 분포')
    st.plotly_chart(fig_hist)

    # 4. 언어별 분포 (languages 리스트 사용)
    st.header('🌍 언어 분포')
    language_counts = pd.Series(df_words['language']).value_counts()
    fig_pie = px.pie(values=language_counts.values, 
                     names=language_counts.index,
                     title='언어별 분포')
    st.plotly_chart(fig_pie)

    # 5. bbox 크기 분포
    st.header('📦 bbox 크기 분포')
    fig_bbox = px.histogram(x=df_words['bbox_size'],
                           title='bbox 크기 분포')
    st.plotly_chart(fig_bbox)

    # 6. 단어 높이/너비 비율 분석
    st.header('📏 단어 크기 분석')
    fig_scatter = px.scatter(df_words, 
                            x='word_width', 
                            y='word_height',
                            title='단어 너비 vs 높이')
    st.plotly_chart(fig_scatter)

if __name__ == '__main__':
    main()