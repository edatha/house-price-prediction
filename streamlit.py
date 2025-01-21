import streamlit as st

# --- PAGE SETUP ---
st.title("Biodata Saya")
col1, col2 = st.columns(2)
with col1:
    st.image("assets/profile_image.png", width=200)  # Ganti dengan path foto Anda
with col2:
    st.subheader("Nama: [Nama Anda]")
    st.subheader("Identitas: [Identitas Anda]")
    st.subheader("Pengalaman: [Pengalaman Anda]")

st.write("---")
st.subheader("Kontak Saya")
st.write("Email: [Email Anda]")
st.write("LinkedIn: [LinkedIn Anda]")
st.write("GitHub: [GitHub Anda]")


