import streamlit.components.v1 as components
import os

def inject_ga():
    GA_ID = os.getenv("GA_TRACKING_ID")
    if not GA_ID:
        return
    components.html(f"""
        <script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){{dataLayer.push(arguments);}}
            gtag('js', new Date());
            gtag('config', '{GA_ID}');
        </script>
    """, height=0)
