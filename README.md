This is the repository for my take-home assignment as part of an interviewing process. It contains a simple Streamlit app which demos the model developed by a fictional colleague. To run it yourself:
1. Fork the repo.
2. Host it on Streamlit Cloud.
3. Set up the OAuth 2.0 environments in Google Cloud and/or Microsoft Azure.
4. Set up the correct secrets. You can follow the official tutorials except that the redirect URI needs to be `redirect_uri = "https://your-app.streamlit.app/oauth2callback"` as opposed to `redirect_uri = "http://localhost:8501/oauth2callback"`. Note the lack of port.
5. Add a line like `allowed_users = ["your@email.com"]` with your e-mail (and other whitelisted e-mails) to the `[auth]` section.

That's it! When expanding on this I recommend breaking up components of the webpage into dedicated files for separation of concerns.