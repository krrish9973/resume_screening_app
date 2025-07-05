import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

#Loading models
clf = pickle.load(open('clf.pkl','rb'))
tf_idf = pickle.load(open('resume_model.pkl', 'rb'))



def clean_resume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

#Web app
def main():
    st.title('Resume Screening App')
    uploaded_file = st.file_uploader("Upload Resume", type=['pdf', 'txt'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            #IF UTF-8 decoding fails, try decoding with Latin - 1
            resume_text = resume_bytes.decode('latin-1')

        clean_resume_text = clean_resume(resume_text)

        #Transform text to vectors
        resume_vector = tf_idf.transform([clean_resume_text])
        prediction_id = clf.predict(resume_vector)[0]
        st.write(prediction_id)

        category_mapping = {14: 'Health and fitness',
                            6: 'Data Science',
                            3: 'Blockchain',
                            9: 'DotNet Developer',
                            19: 'PMO',
                            13: 'Hadoop',
                            2: 'Automation Testing',
                            23: 'Testing',
                            17: 'Network Security Engineer',
                            1: 'Arts',
                            8: 'DevOps Engineer',
                            20: 'Python Developer',
                            22: 'Sales',
                            18: 'Operations Manager',
                            12: 'HR',
                            0: 'Advocate',
                            7: 'Database',
                            5: 'Civil Engineer',
                            21: 'SAP Developer',
                            11: 'Electrical Engineering',
                            15: 'Java Developer',
                            24: 'Web Designing',
                            16: 'Mechanical Engineer',
                            10: 'ETL Developer',
                            4: 'Business Analyst'}
        category_name = category_mapping.get(prediction_id, 'Unknown')
        st.write(f"Predicted Category : {category_name}")


#python main
if __name__ == '__main__':
    main()

