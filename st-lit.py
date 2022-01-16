import streamlit as st
import pandas as pd

table_data = {'Column 1': [1, 2], 'Column 2': [3, 4]}

if st.button('delet dis'):
    del table_data
    st.write('mr button has delet for u')
    
try:
    st.write(pd.DataFrame(data=table_data))
except:
    pass