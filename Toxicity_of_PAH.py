import numpy as np
import pickle
import warnings
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.preprocessing import StandardScaler
import os
from PIL import Image
import streamlit as st
import base64
from io import BytesIO
warnings.filterwarnings("ignore")
# 设置图标和标题
st.set_page_config(page_title="Toxicity of PAHs",  # 设置页面标题
                   page_icon="ico.jpg",    # 设置图标，文件名应与你的.ico文件匹配
                   layout="centered")               # 页面布局（可选），你可以根据需要选择 "wide" 或 "centered"
# 定义计数器
counter_file = 'counter.txt'
if not os.path.exists(counter_file):
    with open(counter_file, 'w') as f:
        f.write('0')
if 'visit_count' not in st.session_state:
    with open(counter_file, 'r') as f:
        visit_count = int(f.read())
    visit_count += 1
    with open(counter_file, 'w') as f:
        f.write(str(visit_count))
    st.session_state.visit_count = visit_count
else:
    visit_count = st.session_state.visit_count

#导入数据
library=pd.read_excel("library.xlsx")
Xtrain=library.iloc[:634,4:-2]
all_compound_info=library['CID'].tolist()+library['SMILES'].tolist()+library['IUPAC Name'].tolist()
Rf_format=pd.read_excel("try.xlsx")
#标准化，质心
std = StandardScaler().fit(Xtrain)
Xtrain_std = pd.DataFrame(std.transform(Xtrain))
centroid_train = pd.DataFrame(np.mean(Xtrain_std, axis=0)).T
centroid_train.columns = Xtrain.columns
#提前储存fps_all
fps_all=[]
for smile in library['SMILES']:
    mol_all = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol_all, 2, nBits=1024)
    fps_all.append(fp)
#设置页面标题
st.markdown(f'<p style="font-size: 35px;color:skyblue;font-weight:bold;background-color:azure;text-align:center;'
    f'padding:20px;border-radius: 10px">Prediction of PAHs Acute Toxicity and <em>RfD</em></p>',unsafe_allow_html=True)
#设置toc图片
image = Image.open("toc.jpg")
buffered = BytesIO()
image.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode()
st.markdown(f"""<div style='text-align: center;'><img src='data:image/jpeg;base64,{img_str}' style='width: 450px;'/></div>""",unsafe_allow_html=True)#调整图片大小

# 创建Tab页
tabs = st.tabs(["Predict", "Search Info", "Help"])
with tabs[1]:
    st.header("Search Info")
    st.dataframe(library.iloc[:,:3], height=600)
with tabs[2]:
    st.header("Help")
    st.write('1. Toxicity category is judged based on EPA rules: LD50 ≤ 50 mg/kg → "High", 50 mg/kg < LD50 ≤ 500 mg/kg → "Moderate", LD50 > 500 mg/kg → "Low".')
    st.write('2. The judgment of AD is based on both Euclidean distance and Tanimoto coefficient of molecular fingerprint.')
    st.write("3. PubChem(https://pubchem.ncbi.nlm.nih.gov/) provides chemical information.")
    st.write("4. If the compound is not in the library or if batch predictions are needed, you can obtain the descriptors from the Chemdes platform (http://www.scbdd.com/chemdes/) and upload the document for toxicity predictions. Please ensure the document follows the format outlined below:")
    st.table(Rf_format)
    st.write("SMILES+MW+descriptors(the number of descriptors can exceed 48, but please ensure the key descriptors are included.)")

with tabs[0]:
    col1, col2= st.columns([1, 2])
    with col1:
        st.markdown("""<div style="display: flex; align-items: center; height: 100%;"><span style="font-size: 17px; color: dark;  position: relative; top: 34px;">
                    PAH or its derivatives</span></div>""",unsafe_allow_html=True)#文本框左侧文字
    with col2:
        input_placeholder = st.empty()
        info_compound=input_placeholder.text_input(label="", value='', key="placeholder",placeholder="Please input SMILES/IUPAC Name/CID")# 用于用户输入内容的文本框
    st.markdown("&nbsp;")

    file_uploader = st.file_uploader("""Upload File (If the compound isn't in the library or batch predictions are needed, a document should be uploaded. Detailed format can be found on the "Help" page.)""", type=["xlsx"])
    #定义predict按钮尺寸
    st.markdown("""<style>.stButton>button {width: 150px;  /* 调整按钮的宽度 */height: 40px;  /* 调整按钮的高度 */font-size: 14px;  /* 调整按钮的字体大小 */
            padding: 5px;  /* 调整按钮的内边距 */display: block; /* 将按钮设置为块级元素，以便于居中 */margin: 0 auto; /* 自动水平居中 */}</style>""",unsafe_allow_html=True)

    with open('model.pkl', 'rb') as f:#读取模型
        clf = pickle.load(f)
    if st.button("Predict", type="primary", use_container_width=True):
        if file_uploader is not None and not info_compound:
            dataval = pd.read_excel(file_uploader)
            AD=[]
            toxicity=[]
            LD50s=[]
            pLD50s=[]
            RfDs=[]
            fps1=[]
            for smiles_val in dataval.iloc[:,0]:
                mol1 = Chem.MolFromSmiles(smiles_val)
                fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
                fps1.append(fp1)
            for i in range(dataval.shape[0]):
                Tanimoto = DataStructs.BulkTanimotoSimilarity(fps1[i], fps_all[:788])
                if max(Tanimoto)==1 :
                    index1=Tanimoto.index(max(Tanimoto))
                    pLD50=library['pLD50'][index1]
                    LD50=library['LD50'][index1]
                    RfD=(round(LD50) / 30000)
                    pLD50s.append(pLD50.round(5))
                    LD50s .append(LD50.round(1))
                    RfDs .append(round(RfD,5))
                    AD.append('Internal data')
                else:
                    preding=(pd.DataFrame(dataval.iloc[i,:][Xtrain.columns]).T).astype(float)
                    pLD50=clf.predict(preding)[0]
                    MW=dataval.iloc[i,1]
                    LD50 =(MW / (10 ** pLD50)) * 1000
                    RfD =(round(LD50) / 30000)
                    pLD50s.append(pLD50.round(5))
                    LD50s .append(LD50.round(0))
                    RfDs .append(round(RfD,5))
                    Xval_std = pd.DataFrame(std.transform(preding))  # 按照训练集的均值和标准差，将待测化合物标准化
                    Xval_std.columns = Xtrain.columns
                    a = np.sqrt(np.sum(np.square(centroid_train.iloc[0, :] - Xval_std.iloc[0, :])))
                    if a <= 10.319 and max(Tanimoto[:634]) >= 0.35:
                        AD.append('In AD')
                    else:
                        AD.append('Out AD')
                if LD50 <= 50:
                    toxicity.append('High')
                elif 50 < LD50 <= 500:
                    toxicity.append('Moderate')
                elif 500 < LD50:
                    toxicity.append('Low')

            o = pd.concat([pd.DataFrame(dataval.iloc[:, 0]), pd.DataFrame(pLD50s), pd.DataFrame(LD50s), pd.DataFrame(RfDs), pd.DataFrame(toxicity), pd.DataFrame(AD)], axis=1)
            o.columns = ['Compound', 'pLD50', 'LD50(mg/kg)', 'RfD(mg/kg-day)', 'EPA Toxicity category', 'In/Out AD']
            st.markdown("""<div style="border-top: 3px solid #87CEEB; margin-bottom: 10px;"></div><h3 style="margin-top: 10px;">Result</h3>""",unsafe_allow_html=True)
            st.dataframe(o)

        elif info_compound and file_uploader is None:
            index=None
            info_str = str(info_compound)  # 将 info_compound 转换为字符串
            if info_compound.isdigit():
                info_compound = int(info_compound)
            if Chem.MolFromSmiles(info_str) is not None:
                mol=Chem.MolFromSmiles(info_str)
                fp_query = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                Tanimoto = DataStructs.BulkTanimotoSimilarity(fp_query, fps_all[:])
                if max(Tanimoto)==1:
                    index=Tanimoto.index(1)
                    smiles_val=library['SMILES'][index]
                elif max(Tanimoto)<1:
                    st.warning("Sorry, the compound is not in the library!")
                    st.markdown("""<div style='text-align: center;'><p style='font-size:13px;'><strong>Please cite:</strong> Wu <i>et al.</i> <i>Environ. Sci. Technol.</i> 2024, 58, 15100−15110 <a href="https://pubs.acs.org/doi/10.1021/acs.est.4c03966?ref=pdf" target="_blank">[paper]</a></div>""",
                                unsafe_allow_html=True)

                    st.markdown(f"""<div style='text-align: right;'><p style='font-size:12px;'>Page views: {visit_count}</p></div>""",
                        unsafe_allow_html=True)
                    st.markdown(f"""<div style='text-align: center;'><p style='font-size:11px;'>If you face any problems or have any valuable suggestions, please contact us (Email: 1417367453@qq.com).</p></div>""",
                        unsafe_allow_html=True)
                    st.stop()

            elif info_compound in all_compound_info:
                index = library.index[library.isin([info_compound]).any(axis=1)].tolist()[0]
                smiles_val = library['SMILES'][index]
            elif info_compound not in all_compound_info:
                st.warning("Sorry, the compound is not in the library!")
                st.markdown("""<div style='text-align: center;'><p style='font-size:13px;'><strong>Please cite:</strong> Wu <i>et al.</i> <i>Environ. Sci. Technol.</i> 2024, 58, 15100−15110 <a href="https://pubs.acs.org/doi/10.1021/acs.est.4c03966?ref=pdf" target="_blank">[paper]</a></div>""",
                            unsafe_allow_html=True)

                st.markdown(f"""<div style='text-align: right;'><p style='font-size:12px;'>Page views: {visit_count}</p></div>""",unsafe_allow_html=True)
                st.markdown(f"""<div style='text-align: center;'><p style='font-size:11px;'>If you face any problems or have any valuable suggestions, please contact us(Email: 1417367453@qq.com).</p></div>""",unsafe_allow_html=True)
                st.stop()

            if index > 787:
                val_compound = pd.DataFrame(library.iloc[index, 4:-2]).T
                val_compound = val_compound[Xtrain.columns].astype(float)
                pLD50 = clf.predict(val_compound)
                mw = library['MW'].to_list()[index]
                LD50 = ((mw / (10 ** pLD50)) * 1000)[0]
                RfD = round(LD50) / 30000
                if LD50 <= 50:
                    toxicity = 'High'
                elif 50 < LD50 <= 500:
                    toxicity = 'Moderate'
                elif 500 < LD50:
                    toxicity = 'Low'

                Xval_std = pd.DataFrame(std.transform(val_compound))  # 按照训练集的均值和标准差，将待测化合物标准化
                Xval_std.columns = Xtrain.columns
                ED = np.sqrt(np.sum(np.square(centroid_train.iloc[0, :] - Xval_std.iloc[0, :])))
                mol1 = Chem.MolFromSmiles(smiles_val)
                # img = Draw.MolToImage(mol1, size=(360, 204))
                img = Draw.rdMolDraw2D(mol1, size=(360, 204))
                img.save('aa.png')
                fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
                Tanimoto = DataStructs.BulkTanimotoSimilarity(fp1, fps_all[:634])

                # 创建两行方框控件
                st.markdown("""
                    <div style="border-top: 3px solid #87CEEB; margin-bottom: 10px;"></div>
                    <h3 style="margin-top: 10px;"
                    """,unsafe_allow_html=True)

                col1, spacer, col2, spacer, col3 = st.columns([1, 0.2, 1, 0.2, 1])
                with col1:
                    st.markdown('<div style="text-align: center;">pLD50</div>', unsafe_allow_html=True)
                    st.markdown(
                        f"""<div style="border: 2px solid black;padding: 10px;border-radius: 5px;background-color: #f0f0f0;text-align: center;">{pLD50[0]:.5f}</div>""",
                        unsafe_allow_html=True)
                    st.markdown("&nbsp;")  # 使用HTML的非断空格实现空行
                    st.markdown('<div style="text-align: center;">Toxicity category</div>', unsafe_allow_html=True)
                    st.markdown(f"""<div style="border: 2px solid black;padding: 10px;border-radius: 5px;background-color: #f0f0f0;text-align: center;">{toxicity}</div>""",
                        unsafe_allow_html=True)

                with col2:
                    st.markdown('<div style="text-align: center;">LD50</div>', unsafe_allow_html=True)
                    st.markdown(f"""<div style="border: 2px solid black;padding: 10px;border-radius: 5px;background-color: #f0f0f0;text-align: center;">{LD50:.0f}</div>""",
                        unsafe_allow_html=True)
                with col3:
                    st.markdown('<div style="text-align: center;">RfD</div>', unsafe_allow_html=True)
                    st.markdown(
                        f"""<div style="border: 2px solid black;padding: 10px;border-radius: 5px;background-color: #f0f0f0;text-align: center;">{RfD:.5f}</div>""",
                        unsafe_allow_html=True)
                    st.markdown("&nbsp;")  # 使用HTML的非断空格实现空行
                    st.markdown('<div style="text-align: center;">In/Out AD</div>', unsafe_allow_html=True)
                    if max(Tanimoto) >= 0.35 and ED <= 10.319:
                        st.markdown(
                            f"""<div style="border: 2px solid black;padding: 10px;border-radius: 5px;background-color: #f0f0f0;text-align: center;">In AD</div>""",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f"""<div style="border: 2px solid black;padding: 10px;border-radius: 5px;background-color: #f0f0f0;text-align: center;">Out AD</div>""",
                            unsafe_allow_html=True)

                col1,col2, col3 = st.columns([0.7,1.2,0.8])
                with col1:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.write('Structure of compound:')
                    st.image("aa.png", width=300)
                with col3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if max(Tanimoto) >= 0.35 and ED <= 10.319:
                        st.image("good.jpg", width=200)  # 替换为你的“good”图片路径
                    else:
                        st.image("bad.jpg", width=200)  # 替换为你的“bad”图片路径
                m = pd.DataFrame({'Compound': [info_compound],'pLD50': [f"{pLD50[0]:.5f}"],'LD50 (mg/kg)': [f"{LD50:.0f}"],'RfD (mg/kg-day)': [f"{RfD:.5f}"],
                    'EPA Toxicity Category': [toxicity],'In/Out AD': 'In AD' if max(Tanimoto) >= 0.35 and ED <= 10.319 else 'Out AD'})
                st.markdown("""<div style="border-top: 3px solid #87CEEB; margin-bottom: 10px;"></div><h3 style="margin-top: 10px;">Result</h3>""",unsafe_allow_html=True)
                st.dataframe(m)

            if index<=787:
                pLD50 = library['pLD50'][index]
                LD50 = library['LD50'][index]
                mol = Chem.MolFromSmiles(smiles_val)
                # img = Draw.MolToImage(mol, size=(360, 204))
                img = Draw.rdMolDraw2D(mol, size=(360, 204))
                img.save('aa.png')
                RfD = round(LD50) / 30000
                if LD50 <= 50:
                    toxicity = 'High'
                elif 50 < LD50 <= 500:
                    toxicity = 'Moderate'
                elif 500 < LD50:
                    toxicity = 'Low'
                st.markdown("""<div style="border-top: 3px solid #87CEEB; margin-bottom: 10px;"></div><h3 style="margin-top: 10px""",unsafe_allow_html=True)

                col1, spacer, col2, spacer, col3 = st.columns([1, 0.2, 1, 0.2, 1])
                with col1:
                    st.markdown('<div style="text-align: center;">pLD50</div>', unsafe_allow_html=True)
                    st.markdown(f"""<div style="border: 2px solid black;padding: 10px;border-radius: 5px;background-color: #f0f0f0;text-align: center;">{pLD50:.5f} (True)</div>""",
                        unsafe_allow_html=True)
                    st.markdown("&nbsp;")  # 使用HTML的非断空格实现空行
                    st.markdown('<div style="text-align: center;">Toxicity category</div>', unsafe_allow_html=True)
                    st.markdown(f"""<div style="border: 2px solid black;padding: 10px;border-radius: 5px;background-color: #f0f0f0;text-align: center;">{toxicity}</div>""",
                        unsafe_allow_html=True)
                with col2:
                    st.markdown('<div style="text-align: center;">LD50</div>', unsafe_allow_html=True)
                    st.markdown(f"""<div style="border: 2px solid black;padding: 10px;border-radius: 5px;background-color: #f0f0f0;text-align: center;">{LD50:.0f}</div>""",
                        unsafe_allow_html=True)
                with col3:
                    st.markdown('<div style="text-align: center;">RfD</div>', unsafe_allow_html=True)
                    st.markdown(f"""<div style="border: 2px solid black;padding: 10px;border-radius: 5px;background-color: #f0f0f0;text-align: center;">{RfD:.5f}</div>""",
                        unsafe_allow_html=True)
                    st.markdown("&nbsp;")  # 使用HTML的非断空格实现空行
                    st.markdown('<div style="text-align: center;">In/Out AD</div>', unsafe_allow_html=True)
                    st.markdown(
                        f"""<div style="border: 2px solid black;padding: 10px;border-radius: 5px;background-color: #f0f0f0;text-align: center;">Internal data</div>""",
                        unsafe_allow_html=True)

                col1,col2, col3 = st.columns([0.7,1.2,0.8])
                with col1:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.write('Structure of compound:')
                    st.image("aa.png", width=300)
                with col3:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.image("true.png", width=200)  # 替换为你的“good”图片路径

                m = pd.DataFrame({'Compound': [info_compound],'pLD50': [f"{pLD50:.5f}"],'LD50 (mg/kg)': [f"{LD50:.0f}"],
                                             'RfD (mg/kg-day)': [f"{RfD:.5f}"],'EPA Toxicity Category': [toxicity],'In/Out AD': 'Internal data'})
                st.markdown("""<div style="border-top: 3px solid #87CEEB; margin-bottom: 10px;"></div><h3 style="margin-top: 10px;">Result</h3>""",unsafe_allow_html=True)

                st.dataframe(m)

        elif file_uploader is None and not info_compound:
            st.warning("Please enter the information of compound or upload a document.")
        elif info_compound and file_uploader is not None:
            st.warning("Please avoid entering compound information and uploading a document simultaneously.")

        st.write('If the compound to be tested is in AD, the predicted value is considered reliable; otherwise, please judge with caution. The low toxicity of the PAH derivatives indicated by the software should be used with caution, as some have been proven to be carcinogenic.')


st.markdown("""<div style='text-align: center;'><p style='font-size:13px;'><strong>Please cite:</strong> Wu <i>et al.</i> <i>Environ. Sci. Technol.</i> 2024, 58, 15100−15110 <a href="https://pubs.acs.org/doi/10.1021/acs.est.4c03966?ref=pdf" target="_blank">[paper]</a></div>""",
            unsafe_allow_html=True)

st.markdown(f"""<div style='text-align: right;'><p style='font-size:12px;'>Page views: {visit_count}</p></div>""",
            unsafe_allow_html=True)
st.markdown(
    f"""<div style='text-align: center;'><p style='font-size:11px;'>If you face any problems or have any valuable suggestions, please contact us(Email: 1417367453@qq.com; xianglei@jnu.edu.cn).</p></div>""",
    unsafe_allow_html=True)
