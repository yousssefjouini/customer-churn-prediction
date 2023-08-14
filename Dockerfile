FROM python:3.9

# Install poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
ENV PATH="${PATH}:/root/.poetry/bin"

# Set working directory
WORKDIR /app

# Copy files
COPY pyproject.toml poetry.lock ./
COPY main.py ./

# Install dependencies
RUN poetry install --no-dev

# Start the application
CMD ["poetry", "run", "streamlit", "run", "app/Home.py"]
