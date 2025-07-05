import streamlit as st
import pickle
import re
import nltk
import PyPDF2
from io import BytesIO

from numpy.f2py.auxfuncs import throw_error

nltk.download('punkt')
nltk.download('stopwords')

#Loading models
clf = pickle.load(open('clf.pkl','rb'))
tf_idf = pickle.load(open('resume_model.pkl', 'rb'))


def extract_text_from_pdf(file_like_object):
    """Extract text from PDF file-like object"""
    text = ""
    try:
        reader = PyPDF2.PdfReader(file_like_object)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:  # Check if text was extracted
                text += page_text + "\n"
    except Exception as e:
        st.error(f"PDF extraction error: {str(e)}")
    return text.strip()

def extract_text_from_txt(file_like_object):
    """Extract text from text file-like object"""
    try:
        return file_like_object.getvalue().decode('utf-8')
    except Exception as e:
        st.error(f"Text extraction error: {str(e)}")
        return ""



def extract_text(uploaded_file):
    """Handle both PDF and text file extraction"""
    if not uploaded_file:
        return ""

    file_extension = uploaded_file.name.split('.')[-1].lower()
    st.write(f"Processing {uploaded_file.name} ({file_extension})")

    try:
        # Reset file pointer to beginning in case it was read before
        uploaded_file.seek(0)

        if file_extension == 'pdf':
            # Create a fresh file-like object from bytes
            return extract_text_from_pdf(BytesIO(uploaded_file.getvalue()))
        elif file_extension == 'txt':
            return extract_text_from_txt(uploaded_file)
        else:
            st.warning(f"Unsupported file type: {file_extension}")
            return ""
    except Exception as e:
        st.error(f"File processing error: {str(e)}")
        return ""


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
        # Extract text from the uploaded file

        try:
            resume_text = extract_text(uploaded_file)
            st.write("Successfully extracted the text from the uploaded resume.")

            # Display extracted text (optional)
            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Make prediction
            st.subheader("Predicted Category")

            # Transform text to vectors
            resume_vector = tf_idf.transform([resume_text])
            prediction_id = clf.predict(resume_vector)[0]

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

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

