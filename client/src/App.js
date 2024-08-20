import logo from './logo.svg';
import './App.css';
import { SocketProvider, SocketConnection, SendData, StartModelTraining } from './components/connections/index.js';


function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <SocketProvider>
          <SocketConnection />
          <SendData />
          <StartModelTraining/>
        </SocketProvider>
      </header>
    </div>
  );
}

export default App;
