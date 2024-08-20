import { io } from "socket.io-client";
import { createContext, useContext, useState, useEffect } from "react";
import JSZip from 'jszip';
import * as ort from 'onnxruntime-web/webgl';

const SocketContext = createContext();
const SOCKET_SERVER_URL = "http://127.0.0.1:5000";

export const useSocket = () => {
    return useContext(SocketContext);
};

export const SocketProvider = ({ children }) => {
    const [socketInstance, setSocketInstance] = useState(null);
    const [webFLClientId, setWebFLClientId] = useState(null);

    useEffect(() => {
        const socket = io(SOCKET_SERVER_URL);

        socket.on('connect', () => {
            console.log('Connecting to the server...');
        });

        socket.on('client_connected', (data) => {
            setWebFLClientId(data.webfl_client_id);
            console.log('You are connected to the server!');
        });

        setSocketInstance(socket);

        return () => {
            socket.disconnect();
        };
    }, []);

    return (
        <SocketContext.Provider value={{socketInstance, webFLClientId}}>
            {children}
        </SocketContext.Provider>
    );
};

export const SocketConnection = () => {
    const socketInstance = useSocket().socketInstance;

    return (
        <div>
            {socketInstance ? "Connected to Server!" : "Connecting..."}
        </div>
    );
};

export const SendData = () => {
    const socketInstance = useSocket().socketInstance;

    const sendDataToServer = () => {
        if (socketInstance) {
            console.log(socketInstance);
        }
    };

    return (
        <button onClick={sendDataToServer}>
            Check Connection Status
        </button>
    );
};

// Utility function to calculate SHA-256 checksum
async function calculateSHA256(arrayBuffer) {
    const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    return hashHex;
}

async function loadTrainingSession() {

    const chkptPath = 'checkpoint';
    const trainingPath = 'training_model.onnx';
    const optimizerPath = 'optimizer_model.onnx';
    const evalPath = 'eval_model.onnx';

    const createOptions = {
        checkpointState: localStorage.getItem(chkptPath),
        trainModel: localStorage.getItem(trainingPath),
        evalModel: localStorage.getItem(evalPath),
        optimizerModel: localStorage.getItem(optimizerPath)
    };

    try {
        const session = await ort.TrainingSession.create(createOptions, {
            executionProviders: ['webgl'] // Specify WebGPU as the execution provider
        });
        return session;
    } catch (err) {
        console.log("Error loading the training session: " + err);
        throw err;
    }
};

const lossNodeName = "onnx::loss::8" // The name of the loss node in the model
const numEpochs = 10 // Number of epochs to train for

// training function for a single epoch
const RunTrainingEpoch = async (session, dataSet, epoch) => {
    const [trainingLosses, setTrainingLosses] = useState([]);

    let batchNum = 0;
    let totalNumBatches = dataSet.numTrainingBatches;
    const epochStartTime = Date.now();
    let iterationsPerSecond = 0;

    for await (const batch of dataSet.trainingBatches) {
        ++batchNum;
        // create input
        const feeds = {
            input: batch.data,
            labels: batch.labels
        }

        // call train step
        const results = await session.runTrainStep(feeds);

        // updating with metrics
        const loss = parseFloat(results[lossNodeName].data);
        setTrainingLosses(losses => losses.concat(loss));
        iterationsPerSecond = batchNum / ((Date.now() - epochStartTime) / 1000);
        const message = `TRAINING | Epoch: ${String(epoch + 1).padStart(2)} | Batch ${String(batchNum).padStart(3)} / ${totalNumBatches} | Loss: ${loss.toFixed(4)} | ${iterationsPerSecond.toFixed(2)} it/s`;
        console.log(message);

        // update weights then reset gradients
        await session.runOptimizerStep();
        await session.lazyResetGrad();
    }
    return [iterationsPerSecond, trainingLosses];
}

export const StartModelTraining = () => {

    const [epochTrainingLosses, setEpochTrainingLosses] = useState([]);
    const [trainingSession, setTrainingSession] = useState(null);
    const socketInstance = useSocket().socketInstance;
    const webFLClientId = useSocket().webFLClientId;

    const getGlobalModel = () => {
        if (socketInstance !== null && socketInstance !== undefined && socketInstance.connected) {
            console.log('Requesting global model from server...');
            socketInstance.emit('request_global_model', { data: socketInstance.id });

            socketInstance.on('global_model_from_server', async (data) => {
                // Decode the Base64 string
                const binaryString = atob(data.data);
                const zipData = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    zipData[i] = binaryString.charCodeAt(i);
                }

                // Calculate checksum of the received ZIP file
                const receivedChecksum = data.checksum;
                const receivedSHA256 = await calculateSHA256(zipData.buffer);

                if (receivedSHA256 === receivedChecksum) {
                    console.log('Checksum matches. ZIP file integrity verified.');
                } else {
                    console.error('Checksum does not match. ZIP file integrity compromised.');
                }

                // Load and process the zip file
                const zip = await JSZip.loadAsync(zipData);

                // Track the promises for storing files
                const fileSavePromises = [];

                // Iterate over the files in the zip
                zip.forEach((relativePath, zipEntry) => {
                    const savePromise = zipEntry.async('uint8array').then(content => {
                        console.log(`Saving file: ${relativePath}`);
                        const base64String = btoa(String.fromCharCode(...content));
                        localStorage.setItem(relativePath, base64String);
                    });
                    fileSavePromises.push(savePromise);
                });

                // Wait for all files to be saved
                await Promise.all(fileSavePromises);

                const trainingSession = await loadTrainingSession();
                setTrainingSession(trainingSession);
                const dataSet = localStorage.getItem(dataSet); // Stored as a json in the local storage

                const startTrainingTime = Date.now();
                console.log('Training started');
                let itersPerSecCumulative = 0;
                for (let epoch = 0; epoch < numEpochs; epoch++) {
                    let res = await RunTrainingEpoch(trainingSession, dataSet, epoch);
                    itersPerSecCumulative += res[0];
                    setEpochTrainingLosses(lossset => lossset.concat(res[1]));
                }
                const trainingTimeMs = Date.now() - startTrainingTime;
                console.log(`Training completed. Total training time: ${trainingTimeMs / 1000} seconds | Average iterations / second: ${(itersPerSecCumulative / numEpochs).toFixed(2)}`);
                
                const payload = {webfl_client_id: webFLClientId, loss: epochTrainingLosses, modelWeights: trainingSession.getContiguousParameters()};
                socketInstance.emit('client_training_completed', { data: payload });
            });

            socketInstance.on("broadcast_global_model", async (data) => {
                const newModelWeights = data.weights;
                await trainingSession.loadParametersBuffer(newModelWeights);

                const dataSet = localStorage.getItem(dataSet); // Stored as a json in the local storage

                const startTrainingTime = Date.now();
                console.log('Training started');
                let itersPerSecCumulative = 0;
                for (let epoch = 0; epoch < numEpochs; epoch++) {
                    let res = await RunTrainingEpoch(trainingSession, dataSet, epoch);
                    itersPerSecCumulative += res[0];
                    setEpochTrainingLosses(lossset => lossset.concat(res[1]));
                }
                const trainingTimeMs = Date.now() - startTrainingTime;
                console.log(`Training completed. Total training time: ${trainingTimeMs / 1000} seconds | Average iterations / second: ${(itersPerSecCumulative / numEpochs).toFixed(2)}`);
                
                const payload = {webfl_client_id: webFLClientId, loss: epochTrainingLosses, modelWeights: trainingSession.getContiguousParameters()};
                socketInstance.emit('client_training_completed', { data: payload });
            });
        }
    }

    return (
        <button onClick={getGlobalModel}>
            Start Training
        </button>
    );
}