import streamlit as st

def main() -> None:
    st.set_page_config(
        page_title="GICES · Test mínimo",
        layout="wide",
    )
    st.title("GICES · Test mínimo")
    st.write("Si ves esta pantalla, el problema de sintaxis está resuelto.")

if __name__ == "__main__":
    main()
