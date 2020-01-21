# Python support can be specified down to the minor or micro version
# (e.g. 3.6 or 3.6.3).
# OS Support also exists for jessie & stretch (slim and full).
# See https://hub.docker.com/r/library/python/ for all supported Python
# tags from Docker Hub.
FROM python:3.7.6-slim-buster

# Récupération fichiers
COPY . /app

# Répertoire de travail
WORKDIR /app

# Using pip:
RUN python3 -m pip install -r requirements.txt

# Déclaration des ports d'entrées à l'app depuis l'extérieur du container
EXPOSE 80
EXPOSE 5000

# Déplacement des fichiers de configuration de streeamlit
# dans un répertoire .streamlit
RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml
WORKDIR /app

# Lance Flask
CMD ["python", "API.py"]

# Lance streamlit
ENTRYPOINT ["streamlit", "run"]
CMD ["DASHBOARD.py"]