FROM python:3.9

# Instalar dependencias
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copiar el código de la aplicación
COPY . /app

# Establecer el directorio de trabajo
WORKDIR /app

# Comando por defecto
CMD ["bash"]
