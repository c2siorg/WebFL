import logo from './logo.svg';
import './App.css';

import ModelTrainer from './components/modelTrainer';
import * as tf from "@tensorflow/tfjs"
tf.setBackend('webgl');

function App() {
  return (
    <div className="App">
      <ModelTrainer />
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
