# WebFL
WebGPU for Federated Learning

## Overview
This project is a web application built with React and a Python backend. The project structure includes both frontend and backend components, with dependencies managed via Node.js and Python virtual environments.

## Project Structure
- **Frontend**: The React application is located in the `/client` directory.
- **Backend**: The Python server is located in the `/server` directory.

## Setup Instructions

### Prerequisites
- Node.js (v18+) and npm (or yarn)
- Python (tested on v3.10) and virtualenv

### Frontend Setup
1. **Install dependencies**:
    ```sh
    cd client & npm install
    # or
    cd client & yarn install
    ```

2. **Start the development server**:
    ```sh
    npm start
    # or
    yarn start
    ```

### Backend Setup
1. **Navigate to the server directory**:
    ```sh
    cd server
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment**:
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```
    - On Windows:
        ```sh
        .\venv\Scripts\activate
        ```

4. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

5. **Start the Python server**:
    ```sh
    flask run
    ```