import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const emptyCsv = {
    name: '',
    headers: [],
    rows: [],
    rawRows: [],
    results: [],
    resultHeader: '',
    attributeRanges: [],
    error: '',
};

function parseCsvLine(line) {
    const values = [];
    let current = '';
    let quoted = false;

    for (let index = 0; index < line.length; index += 1) {
        const char = line[index];
        const next = line[index + 1];

        if (char === '"' && quoted && next === '"') {
            current += '"';
            index += 1;
        } else if (char === '"') {
            quoted = !quoted;
        } else if (char === ',' && !quoted) {
            values.push(current.trim());
            current = '';
        } else {
            current += char;
        }
    }

    values.push(current.trim());
    return values;
}

function parseCsvInputRow(row) {
    return row.map((value) => Number(value));
}

function getAttributeRanges(rows) {
    const inputCount = rows[0]?.length ?? 0;

    return Array.from({ length: inputCount }, (_, attributeIndex) => {
        const values = rows.map((row) => row[attributeIndex]);

        return {
            min: Math.min(...values),
            max: Math.max(...values),
        };
    });
}

function normalizeCsvRows(rows, attributeRanges) {
    return rows.map((row) =>
        row.map((value, attributeIndex) => {
            const range = attributeRanges[attributeIndex];

            if (!range || range.max === range.min) {
                return 0;
            }

            return (value - range.min) / (range.max - range.min);
        })
    );
}

function parseCsv(text) {
    const lines = text
        .replace(/^\uFEFF/, '')
        .split(/\r\n|\n|\r/)
        .filter((line) => line.trim().length > 0);

    if (lines.length === 0) {
        throw new Error('CSV vazio.');
    }

    const allHeaders = parseCsvLine(lines[0]);
    const resultHeader = allHeaders.at(-1) ?? '';
    const headers = allHeaders.slice(0, -1);
    const parsedRows = lines.slice(1).map(parseCsvLine);
    const rawRows = parsedRows.map((row) => parseCsvInputRow(row.slice(0, -1)));
    const results = parsedRows.map((row) => row.at(-1) ?? '');

    if (rawRows.some((row) => row.some((value) => Number.isNaN(value)))) {
        throw new Error('CSV possui valores de entrada nao numericos.');
    }

    return { headers, rawRows, results, resultHeader };
}

function getUniqueResults(results) {
    return [...new Set(results.filter((result) => result !== undefined && result !== ''))];
}

function App() {
    const [mode, setMode] = useState('separate');
    const [trainCsv, setTrainCsv] = useState(emptyCsv);
    const [testCsv, setTestCsv] = useState(emptyCsv);
    const [singleCsv, setSingleCsv] = useState(emptyCsv);
    const [singleCsvSource, setSingleCsvSource] = useState(null);
    const [trainSplitPercent, setTrainSplitPercent] = useState(70);
    const [numberOfHiddenNeurons, setNumberOfHiddenNeurons] = useState(0);
    const [numberOfOutputs, setNumberOfOutputs] = useState(0);
    const [trainingUniqueResults, setTrainingUniqueResults] = useState([]);
    const [config, setConfig] = useState({
        hiddenNeurons: '',
        maxEpochs: '',
        errorThreshold: '',
        learningRate: '',
        activationFunction: '',
    });
    const [mlp, setMlp] = useState({
        // numberOfHiddenNeurons
        // numberOfOutputs
        // mlpHiddenInformation
        // mlpOutputInformation
        // mlpModel
        avgError: 0,
    });
    const [trainingState, setTrainingState] = useState({
        status: 'idle',
        epochs: 0,
        avgError: 0,
        history: [],
        plateau: false,
        stoppedBy: '',
    });
    const [evaluationState, setEvaluationState] = useState({
        status: 'idle',
        total: 0,
        correct: 0,
        accuracy: 0,
        labels: [],
        matrix: [],
        predictions: [],
    });
    const [predictTest, setPredictTest] = useState({
        lineIndex: 0,
        inputs: [],
        expectedResult: '',
        predictedResult: '',
        correct: null,
        outputs: [],
        result: null,
        error: '',
    });

    const [mlpHiddenInformation, setMlpHiddenInformation] = useState({
        net: [],
        erro: [],
        i: [],
    });
    const [mlpOutputInformation, setMlpOutputInformation] = useState({
        net: [],
        erro: [],
        i: [],
    });

    const [mlpModel, setMlpModel] = useState({
        inputToHidden: [],
        hiddenToOutput: [],
    });
    const [previousMlpModel, setPreviousMlpModel] = useState(null);
    const [weights, setWeights] = useState([]);
    const [notification, setNotification] = useState(null);
    const [isNotificationHovered, setIsNotificationHovered] = useState(false);
    const activeCsv = mode === 'single' ? singleCsv : trainCsv;
    const testRows = testCsv.rows.length;
    const inputCount = activeCsv.headers.length;

    // log para desenvolvimento: exibe o conteúdo do CSV no console ao carregar
    React.useEffect(() => {
        if (trainCsv.name) {
            console.log('Conteúdo do CSV de treino:', trainCsv);
        }
        if (testCsv.name) {
            console.log('Conteúdo do CSV de teste:', testCsv);
        }
        if (singleCsv.name) {
            console.log('Conteúdo do CSV único:', singleCsv);
        }
        if (config.hiddenNeurons || config.maxEpochs || config.errorThreshold || config.learningRate || config.activationFunction) {
            console.log('Configuração:', config);
        }
        if (trainingUniqueResults.length > 0) {
            console.log('Resultados únicos do treinamento:', trainingUniqueResults);
        }
    }, [trainCsv, testCsv, singleCsv, config, trainingUniqueResults]);

    React.useEffect(() => {
        console.log('Pesos gerados:', weights);
    }, [weights]);

    React.useEffect(() => {
        console.log('Modelo MLP atualizado:', mlpModel);
    }, [mlpModel]);

    React.useEffect(() => {
        if (!notification || isNotificationHovered) {
            return undefined;
        }

        const timeoutId = window.setTimeout(() => {
            setNotification(null);
        }, 5200);

        return () => window.clearTimeout(timeoutId);
    }, [notification, isNotificationHovered]);

    async function readCsvFile(file, setter, shouldStoreTrainingResults = false) {
        if (!file) {
            setter(emptyCsv);
            if (shouldStoreTrainingResults) {
                setTrainingUniqueResults([]);
                setTestCsv(emptyCsv);
            }
            return;
        }

        try {
            const text = await file.text();
            console.log('CSV lido bruto:', {
                name: file.name,
                size: file.size,
                textLength: text.length,
                firstLine: text.split(/\r\n|\n|\r/)[0] ?? '',
            });

            const parsed = parseCsv(text);
            const attributeRanges = shouldStoreTrainingResults ? getAttributeRanges(parsed.rawRows) : trainCsv.attributeRanges;

            if (!shouldStoreTrainingResults && attributeRanges.length === 0) {
                throw new Error('Carregue o CSV de treino antes do CSV de teste.');
            }

            const rows = normalizeCsvRows(parsed.rawRows, attributeRanges);
            setter({
                name: file.name,
                headers: parsed.headers,
                rows,
                rawRows: parsed.rawRows,
                results: parsed.results,
                resultHeader: parsed.resultHeader,
                attributeRanges,
                error: '',
            });
            if (shouldStoreTrainingResults) {
                const uniqueTrainingResults = getUniqueResults(parsed.results);
                setTrainingUniqueResults(uniqueTrainingResults);
                setNumberOfOutputs(uniqueTrainingResults.length);
                setTestCsv((current) => {
                    if (current.rawRows.length === 0) {
                        return current;
                    }

                    return {
                        ...current,
                        rows: normalizeCsvRows(current.rawRows, attributeRanges),
                        attributeRanges,
                        error: '',
                    };
                });
                console.log('Classes únicas do treinamento:', uniqueTrainingResults);
            }
        } catch (error) {
            console.error('Falha ao ler CSV:', error);
            setter({
                ...emptyCsv,
                name: file.name,
                error: 'Não foi possível ler o CSV selecionado.',
            });
            if (shouldStoreTrainingResults) {
                setTrainingUniqueResults([]);
            }
        }
    }

    async function readSingleCsvFile(file) {
        if (!file) {
            setSingleCsv(emptyCsv);
            setTestCsv(emptyCsv);
            setSingleCsvSource(null);
            setTrainingUniqueResults([]);
            return;
        }

        try {
            const text = await file.text();
            const parsed = parseCsv(text);
            if (parsed.rawRows.length < 2) {
                throw new Error('Arquivo unico precisa de pelo menos duas linhas de dados.');
            }
            const nextSource = { name: file.name, ...parsed };
            setSingleCsvSource(nextSource);
            applySingleCsvSplit(nextSource, trainSplitPercent);
        } catch (error) {
            console.error('Falha ao dividir CSV:', error);
            setSingleCsv({
                ...emptyCsv,
                name: file.name,
                error: 'Nao foi possivel dividir o CSV selecionado.',
            });
            setTestCsv(emptyCsv);
            setSingleCsvSource(null);
            setTrainingUniqueResults([]);
        }
    }

    function applySingleCsvSplit(source, splitPercent) {
        const safePercent = Math.min(90, Math.max(10, Number(splitPercent) || 70));
        const splitRatio = safePercent / 100;
        const splitIndex = Math.min(source.rawRows.length - 1, Math.max(1, Math.floor(source.rawRows.length * splitRatio)));
        const trainingRawRows = source.rawRows.slice(0, splitIndex);
        const trainingResults = source.results.slice(0, splitIndex);
        const testingRawRows = source.rawRows.slice(splitIndex);
        const testingResults = source.results.slice(splitIndex);
        const attributeRanges = getAttributeRanges(trainingRawRows);
        const uniqueTrainingResults = getUniqueResults(trainingResults);

        setTrainSplitPercent(safePercent);
        setSingleCsv({
            name: `${source.name} (treino ${safePercent}%)`,
            headers: source.headers,
            rows: normalizeCsvRows(trainingRawRows, attributeRanges),
            rawRows: trainingRawRows,
            results: trainingResults,
            resultHeader: source.resultHeader,
            attributeRanges,
            error: '',
        });
        setTestCsv({
            name: `${source.name} (teste ${100 - safePercent}%)`,
            headers: source.headers,
            rows: normalizeCsvRows(testingRawRows, attributeRanges),
            rawRows: testingRawRows,
            results: testingResults,
            resultHeader: source.resultHeader,
            attributeRanges,
            error: testingRawRows.length === 0 ? 'Arquivo sem linhas suficientes para teste.' : '',
        });
        setTrainingUniqueResults(uniqueTrainingResults);
        setNumberOfOutputs(uniqueTrainingResults.length);
        setEvaluationState((current) => ({
            ...current,
            status: 'idle',
            total: 0,
            correct: 0,
            accuracy: 0,
            labels: [],
            matrix: [],
            predictions: [],
        }));
        setNotification({
            title: 'Arquivo dividido',
            description: 'O CSV completo foi separado automaticamente em treino e teste.',
            details: [
                `${trainingRawRows.length} linhas de treino`,
                `${testingRawRows.length} linhas de teste`,
                `${safePercent}% / ${100 - safePercent}%`,
            ],
        });
    }

    function handleTrainSplitPercentChange(event) {
        const nextPercent = Number(event.target.value);
        setTrainSplitPercent(nextPercent);

        if (singleCsvSource) {
            applySingleCsvSplit(singleCsvSource, nextPercent);
        }
    }

    function handleHiddenNeuronsRecommendation() {
        const csv = mode === 'single' ? singleCsv : trainCsv;

        if (csv.headers.length === 0) {
            setNotification({
                title: 'CSV necessário',
                description: 'Carregue um CSV antes de solicitar a recomendação.',
            });
            return;
        }

        const numberOfInputs = csv.headers.length;
        const numberOfOutputs = trainingUniqueResults.length || getUniqueResults(csv.results).length;
        const hiddenNeurons = recomendationNumberHiddenNeurons({
            numberOfInputs,
            numberOfOutputs,
        });
        setNumberOfHiddenNeurons(hiddenNeurons);
        setNumberOfOutputs(numberOfOutputs);
        setConfig({ ...config, hiddenNeurons: String(hiddenNeurons) });
        setNotification({
            title: 'Recomendação aplicada',
            description: `O campo recebeu ${hiddenNeurons} neurônios ocultos com base no CSV ${csv.name}.`,
            formula: `(${numberOfInputs} entradas + ${numberOfOutputs} saídas) / 2`,
            result: hiddenNeurons,
            details: [`${numberOfInputs} colunas de entrada`, `${numberOfOutputs} classes de saída`, `${csv.rows.length} linhas lidas`],
        });
    }

    function handleRandomModelTest() {
        const csv = mode === 'single' ? singleCsv : trainCsv;
        const numberOfInputs = csv.headers.length > 0 ? csv.headers.length : 3;
        const numberOfOutputsFromCsv = trainingUniqueResults.length || getUniqueResults(csv.results).length;
        const numberOfOutputs = numberOfOutputsFromCsv || 2;
        const hiddenNeuronsCount = Number.parseInt(config.hiddenNeurons, 10) || 2;
        const randomWeights = generateRandomWeights(numberOfInputs, hiddenNeuronsCount, numberOfOutputs);
        const { MLPWeights, MlpInformation } = buildMlpWeights(numberOfInputs, hiddenNeuronsCount, numberOfOutputs, randomWeights, trainingUniqueResults);

        setWeights(randomWeights);
        setMlpHiddenInformation(MlpInformation.hidden);
        setMlpOutputInformation(MlpInformation.output);
        setPreviousMlpModel(null);
        setMlpModel(MLPWeights);
        setNotification({
            title: 'Teste aleatório gerado',
            description: 'Um conjunto de pesos aleatórios foi criado apenas para visualizar a estrutura do modelo.',
            details: [
                `${numberOfInputs} entradas`,
                `${hiddenNeuronsCount} neurônios ocultos`,
                `${numberOfOutputs} saídas`,
                `${randomWeights.length} pesos aleatórios`,
            ],
        });
    }

    function handlePrepareTrainingLayout() {
        const csv = mode === 'single' ? singleCsv : trainCsv;

        if (csv.results.length === 0) {
            setNotification({
                title: 'CSV necessário',
                description: 'Carregue uma base de treino antes de preparar a etapa de treinamento.',
            });
            return;
        }

        const uniqueTrainingResults = getUniqueResults(csv.results);
        setTrainingUniqueResults(uniqueTrainingResults);
        setNumberOfOutputs(uniqueTrainingResults.length);
        setNotification({
            title: 'Treinamento preparado',
            description: 'As classes únicas da base de treino foram armazenadas no estado da etapa de treinamento.',
            details: [`${uniqueTrainingResults.length} classes únicas`, `${csv.rows.length} linhas de treino`],
        });
        console.log('Classes únicas do treinamento:', uniqueTrainingResults);
    }

    function handleTrainMlp() {
        const csv = mode === 'single' ? singleCsv : trainCsv;
        const uniqueTrainingResults = trainingUniqueResults.length > 0 ? trainingUniqueResults : getUniqueResults(csv.results);
        const hiddenNeuronsCount = Number.parseInt(config.hiddenNeurons, 10);
        const maxEpochs = Number.parseInt(config.maxEpochs, 10);
        const errorThreshold = Number(config.errorThreshold);
        const learningRate = Number(config.learningRate);

        if (csv.rows.length === 0 || csv.results.length === 0) {
            setNotification({
                title: 'CSV necessario',
                description: 'Carregue a base de treino antes de iniciar o treinamento.',
            });
            return;
        }

        if (!hiddenNeuronsCount || hiddenNeuronsCount <= 0 || !maxEpochs || maxEpochs <= 0) {
            setNotification({
                title: 'Configuracao incompleta',
                description: 'Informe neuronios ocultos e numero maximo de epocas maiores que zero.',
            });
            return;
        }

        if (Number.isNaN(errorThreshold) || errorThreshold < 0) {
            setNotification({
                title: 'Limiar invalido',
                description: 'Informe um limiar de erro maior ou igual a zero.',
            });
            return;
        }

        if (Number.isNaN(learningRate) || learningRate <= 0 || learningRate > 1) {
            setNotification({
                title: 'Taxa invalida',
                description: 'A taxa de aprendizagem deve ser maior que 0 e menor ou igual a 1.',
            });
            return;
        }

        if (!config.activationFunction) {
            setNotification({
                title: 'Funcao necessaria',
                description: 'Selecione Linear, Logistica ou Tangente hiperbolica.',
            });
            return;
        }

        runTraining({
            csv,
            uniqueTrainingResults,
            hiddenNeuronsCount,
            maxEpochs,
            errorThreshold,
            nextConfig: config,
            modeLabel: 'Treinamento',
        });
    }

    function runTraining({ csv, uniqueTrainingResults, hiddenNeuronsCount, maxEpochs, errorThreshold, nextConfig, modeLabel, initialState }) {
        const previousModel = mlpModel.inputToHidden.length > 0 ? mlpModel : null;
        const trainingResult = trainMlp({
            rows: csv.rows,
            results: csv.results,
            uniqueResults: uniqueTrainingResults,
            hiddenNeuronsCount,
            maxEpochs,
            errorThreshold,
            config: nextConfig,
            initialState,
        });

        setTrainingUniqueResults(uniqueTrainingResults);
        setNumberOfOutputs(uniqueTrainingResults.length);
        setWeights(trainingResult.weights);
        setMlpHiddenInformation(trainingResult.hidden);
        setMlpOutputInformation(trainingResult.output);
        setPreviousMlpModel(previousModel);
        setMlpModel(trainingResult.model);
        setMlp((current) => ({
            ...current,
            avgError: trainingResult.avgError,
        }));
        setTrainingState({
            status: 'trained',
            epochs: trainingResult.epochs,
            avgError: trainingResult.avgError,
            history: trainingResult.history,
            plateau: trainingResult.plateau,
            stoppedBy: trainingResult.stoppedBy,
        });
        setNotification({
            title: `${modeLabel} concluido`,
            description: `${modeLabel} encerrado por ${trainingResult.stoppedBy}.`,
            details: [
                `${trainingResult.epochs} epocas acumuladas`,
                `erro medio ${trainingResult.avgError.toFixed(6)}`,
                `${trainingResult.weights.length} pesos ajustados`,
            ],
        });
        console.log(`${modeLabel} MLP:`, trainingResult);
    }

    function handleContinueTraining(shouldReduceLearningRate = false) {
        const csv = mode === 'single' ? singleCsv : trainCsv;
        const uniqueTrainingResults = trainingUniqueResults.length > 0 ? trainingUniqueResults : getUniqueResults(csv.results);
        const hiddenNeuronsCount = Number.parseInt(config.hiddenNeurons, 10);
        const maxEpochs = Number.parseInt(config.maxEpochs, 10);
        const errorThreshold = Number(config.errorThreshold);
        const currentLearningRate = Number(config.learningRate);

        if (mlpModel.inputToHidden.length === 0 || mlpHiddenInformation.net.length === 0 || mlpOutputInformation.net.length === 0) {
            setNotification({
                title: 'Modelo necessario',
                description: 'Treine a MLP pelo menos uma vez antes de continuar.',
            });
            return;
        }

        if (!hiddenNeuronsCount || hiddenNeuronsCount <= 0 || !maxEpochs || maxEpochs <= 0 || Number.isNaN(errorThreshold)) {
            setNotification({
                title: 'Configuracao incompleta',
                description: 'Confira neuronios ocultos, epocas e limiar de erro antes de continuar.',
            });
            return;
        }

        if (Number.isNaN(currentLearningRate) || currentLearningRate <= 0 || currentLearningRate > 1) {
            setNotification({
                title: 'Taxa invalida',
                description: 'A taxa de aprendizagem deve ser maior que 0 e menor ou igual a 1.',
            });
            return;
        }

        const nextLearningRate = shouldReduceLearningRate ? Math.max(currentLearningRate * 0.9, 0.000001) : currentLearningRate;
        const nextConfig = {
            ...config,
            learningRate: String(nextLearningRate),
        };

        if (shouldReduceLearningRate) {
            setConfig(nextConfig);
        }

        runTraining({
            csv,
            uniqueTrainingResults,
            hiddenNeuronsCount,
            maxEpochs,
            errorThreshold,
            nextConfig,
            modeLabel: shouldReduceLearningRate ? 'Continuacao com taxa reduzida' : 'Continuacao',
            initialState: {
                model: mlpModel,
                hidden: mlpHiddenInformation,
                output: mlpOutputInformation,
                history: trainingState.history,
            },
        });
    }

    function handlePredictTest() {
        const csv = testCsv.rows.length > 0 ? testCsv : mode === 'single' ? singleCsv : trainCsv;
        const lineIndex = Number.parseInt(predictTest.lineIndex, 10) || 0;

        if (mlpModel.inputToHidden.length === 0 || mlpModel.hiddenToOutput.length === 0) {
            setNotification({
                title: 'Modelo necessário',
                description: 'Treine a MLP antes de testar a inferencia.',
            });
            return;
        }

        if (!csv.rows[lineIndex] || !csv.results[lineIndex]) {
            setPredictTest((current) => ({
                ...current,
                error: 'Linha inválida para testar o predict.',
            }));
            setNotification({
                title: 'Linha inválida',
                description: 'Carregue uma base de teste e use um indice de linha existente.',
            });
            return;
        }

        const lineInputs = csv.rows[lineIndex].map((value) => Number(value));
        const expectedResult = csv.results[lineIndex];
        const predictionResult = runMlp(mlpModel, lineInputs, config);
        const correct = expectedResult === predictionResult.predictedLabel;
        setPredictTest({
            lineIndex,
            inputs: lineInputs,
            expectedResult,
            predictedResult: predictionResult.predictedLabel,
            correct,
            outputs: predictionResult.outputs,
            result: predictionResult,
            error: '',
        });
        console.log('Teste predict:', {
            lineIndex,
            lineInputs,
            expectedResult,
            predictedResult: predictionResult.predictedLabel,
            result: predictionResult,
        });
        setNotification({
            title: 'Predict testado',
            description: correct ? 'A classe obtida bateu com a esperada.' : 'A classe obtida foi diferente da esperada.',
            details: [
                `linha ${lineIndex + 1}`,
                `classe esperada ${expectedResult}`,
                `classe obtida ${predictionResult.predictedLabel}`,
                `${predictionResult.outputs.length} saidas calculadas`,
            ],
        });
    }

    function handleEvaluateTestSet() {
        const csv = testCsv;
        const labels = trainingUniqueResults.length > 0 ? trainingUniqueResults : getUniqueResults(trainCsv.results);

        if (mlpModel.inputToHidden.length === 0 || mlpOutputInformation.classe.length === 0) {
            setNotification({
                title: 'Modelo necessario',
                description: 'Treine a MLP antes de avaliar a base de teste.',
            });
            return;
        }

        if (csv.rows.length === 0 || csv.results.length === 0) {
            setNotification({
                title: 'CSV de teste necessario',
                description: 'Carregue uma base de teste para montar a matriz de confusao.',
            });
            return;
        }

        const evaluation = evaluateMlp({
            rows: csv.rows,
            results: csv.results,
            labels,
            model: mlpModel,
            config,
        });

        setEvaluationState({
            status: 'evaluated',
            ...evaluation,
        });
        setNotification({
            title: 'Teste avaliado',
            description: 'A matriz de confusao foi calculada com a base de teste.',
            details: [
                `${evaluation.correct}/${evaluation.total} acertos`,
                `${(evaluation.accuracy * 100).toFixed(2)}% de acuracia`,
            ],
        });
        console.log('Avaliacao da base de teste:', evaluation);
    }

    return (
        <main className="app-shell">
            <header className="topbar">
                <div>
                    <p className="eyebrow">Trabalho de IA - 2º Bimestre</p>
                    <h1>Ferramenta MLP Backpropagation</h1>
                </div>
                <span className="status-pill">Layout sem regras de negócio</span>
            </header>

            <section className="workspace">
                <aside className="sidebar">
                    <SectionTitle title="Entrada de Dados" subtitle="CSV carregado localmente no navegador" />

                    <div className="segmented" aria-label="Modo de entrada">
                        <button className={mode === 'separate' ? 'active' : ''} onClick={() => setMode('separate')}>
                            Treino + Teste
                        </button>
                        <button className={mode === 'single' ? 'active' : ''} onClick={() => setMode('single')}>
                            Arquivo único
                        </button>
                    </div>

                    {mode === 'separate' ? (
                        <div className="stack">
                            <FileInput label="CSV de treino" csv={trainCsv} onFile={(file) => readCsvFile(file, setTrainCsv, true)} />
                            <FileInput label="CSV de teste" csv={testCsv} onFile={(file) => readCsvFile(file, setTestCsv)} />
                        </div>
                    ) : (
                        <div className="stack">
                            <FileInput label="CSV completo" csv={singleCsv} onFile={readSingleCsvFile} />
                            <label className="field">
                                <span>Percentual para treino</span>
                                <input
                                    type="number"
                                    min="10"
                                    max="90"
                                    value={trainSplitPercent}
                                    onChange={handleTrainSplitPercentChange}
                                />
                            </label>
                            <div className="notice">
                                O arquivo completo e dividido automaticamente em {trainSplitPercent}% treino e {100 - trainSplitPercent}% teste.
                            </div>
                        </div>
                    )}

                    <SectionTitle title="Configuração" subtitle="Campos visuais para futura implementação" />

                    <div className="field">
                        <span>Neurônios na camada escondida</span>
                        <div className="input-action">
                            <input
                                type="number"
                                min="1"
                                placeholder="Ex.: 5"
                                value={config.hiddenNeurons}
                                onChange={(e) => setConfig({ ...config, hiddenNeurons: e.target.value })}
                            />
                            <button type="button" onClick={handleHiddenNeuronsRecommendation}>
                                Recomendado
                            </button>
                        </div>
                    </div>

                    <label className="field">
                        <span>Número máximo de épocas</span>
                        <input
                            type="number"
                            min="1"
                            placeholder="Ex.: 1000"
                            value={config.maxEpochs}
                            onChange={(e) => setConfig({ ...config, maxEpochs: e.target.value })}
                        />
                    </label>

                    <label className="field">
                        <span>Limiar de erro</span>
                        <input
                            type="text"
                            placeholder="Ex.: 0.001"
                            value={config.errorThreshold}
                            onChange={(e) => setConfig({ ...config, errorThreshold: e.target.value })}
                        />
                    </label>

                    <label className="field">
                        <span>Taxa de aprendizagem</span>
                        <input
                            type="text"
                            placeholder="0 < N <= 1"
                            value={config.learningRate}
                            onChange={(e) => setConfig({ ...config, learningRate: e.target.value })}
                        />
                    </label>

                    <label className="field">
                        <span>Função de transferência</span>
                        <select value={config.activationFunction} onChange={(e) => setConfig({ ...config, activationFunction: e.target.value })}>
                            <option value="" disabled>
                                Selecione
                            </option>
                            <option>Linear</option>
                            <option>Logística</option>
                            <option>Tangente hiperbólica</option>
                        </select>
                    </label>

                    <div className="button-row">
                        <button className="primary" type="button" onClick={handlePrepareTrainingLayout}>
                            Preparar layout
                        </button>
                        <button className="primary" type="button" onClick={handleTrainMlp}>
                            Treinar MLP
                        </button>
                        <button
                            className="ghost"
                            type="button"
                            onClick={() => {
                                setConfig({
                                    hiddenNeurons: '',
                                    maxEpochs: '',
                                    errorThreshold: '',
                                    learningRate: '',
                                    activationFunction: '',
                                });
                                setNotification({
                                    title: 'Configuração limpa',
                                    description: 'Os campos visuais de configuração foram limpos.',
                                });
                            }}
                        >
                            Limpar
                        </button>
                    </div>
                </aside>

                <section className="content">
                    <section className="summary-strip" aria-label="Resumo dos dados">
                        <SummaryItem label="Arquivo ativo" value={activeCsv.name || 'Nenhum CSV'} />
                        <SummaryItem label="Entradas" value={inputCount || '-'} />
                        <SummaryItem label="Linhas de treino" value={activeCsv.rows.length || '-'} />
                        <SummaryItem label="Linhas de teste" value={testRows || '-'} />
                    </section>

                    <div className="grid two">
                        <CsvPanel title="Base de Treino" csv={mode === 'single' ? singleCsv : trainCsv} />
                        <CsvPanel title="Base de Teste" csv={testCsv} />
                    </div>

                    <section className="panel network-panel">
                        <div className="panel-heading">
                            <SectionTitle title="Arquitetura da Rede" subtitle="Representação visual sem treino ou inferência" />
                            <button className="soft-action" type="button" onClick={handleRandomModelTest}>
                                Testar pesos aleatórios
                            </button>
                            <button className="soft-action" type="button" onClick={handlePredictTest}>
                                Testar predict
                            </button>
                            <button className="soft-action" type="button" onClick={handleEvaluateTestSet}>
                                Avaliar teste
                            </button>
                        </div>
                        <NetworkGraph
                            model={mlpModel}
                            inputLabels={activeCsv.headers.length > 0 ? activeCsv.headers : ['X1', 'X2', 'X3']}
                            hiddenCount={Number.parseInt(config.hiddenNeurons, 10) || mlpHiddenInformation.net.length || 3}
                            outputLabels={trainingUniqueResults.length > 0 ? trainingUniqueResults : ['C1', 'C2', 'C3']}
                        />
                        <ModelPreview model={mlpModel} previousModel={previousMlpModel} weights={weights} />
                        <PredictTestPanel predictTest={predictTest} setPredictTest={setPredictTest} avgError={mlp.avgError} />
                    </section>

                    <div className="grid two">
                        <TrainingPanel
                            uniqueResults={trainingUniqueResults}
                            trainingState={trainingState}
                            onContinueTraining={() => handleContinueTraining(false)}
                            onReduceLearningRate={() => handleContinueTraining(true)}
                        />
                        <ConfusionMatrixPanel evaluationState={evaluationState} />
                    </div>
                </section>
            </section>
            {notification ? (
                <div
                    className="toast"
                    role="status"
                    aria-live="polite"
                    onMouseEnter={() => setIsNotificationHovered(true)}
                    onMouseLeave={() => setIsNotificationHovered(false)}
                >
                    <div className="toast-mark">i</div>
                    <div className="toast-content">
                        <strong>{notification.title}</strong>
                        <span>{notification.description}</span>
                        {notification.formula ? (
                            <div className="toast-formula">
                                <span>Fórmula</span>
                                <code>
                                    {notification.formula} = {notification.result}
                                </code>
                            </div>
                        ) : null}
                        {notification.details ? (
                            <div className="toast-details">
                                {notification.details.map((detail) => (
                                    <span key={detail}>{detail}</span>
                                ))}
                            </div>
                        ) : null}
                    </div>
                    <button type="button" onClick={() => setNotification(null)} aria-label="Fechar notificação">
                        ×
                    </button>
                </div>
            ) : null}
        </main>
    );
}

function SectionTitle({ title, subtitle }) {
    return (
        <div className="section-title">
            <h2>{title}</h2>
            {subtitle ? <p>{subtitle}</p> : null}
        </div>
    );
}

function SummaryItem({ label, value }) {
    return (
        <div className="summary-item">
            <span>{label}</span>
            <strong>{value}</strong>
        </div>
    );
}

function FileInput({ label, csv, onFile }) {
    return (
        <label className="file-input">
            <span>{label}</span>
            <div className={csv.name ? 'file-drop has-file' : 'file-drop'}>
                <strong>{csv.name || 'Selecionar arquivo CSV'}</strong>
                <small>{csv.name ? `${csv.rows.length} linhas lidas` : 'Clique para carregar do computador'}</small>
            </div>
            <input type="file" accept=".csv,text/csv" onChange={(event) => onFile(event.target.files?.[0])} />
        </label>
    );
}

function CsvPanel({ title, csv }) {
    const hasData = csv.headers.length > 0;
    const previewRows = csv.rows.slice(0, 6);

    return (
        <section className="panel">
            <SectionTitle title={title} subtitle={csv.name || 'Nenhum arquivo carregado'} />
            {csv.error ? <div className="error">{csv.error}</div> : null}
            {hasData ? (
                <>
                    <div className="metrics">
                        <span>{csv.headers.length} entradas</span>
                        <span>{csv.resultHeader || 'resultado'}</span>
                        <span>{csv.rows.length} linhas lidas</span>
                    </div>
                    <div className="table-wrap">
                        <table>
                            <thead>
                                <tr>
                                    {csv.headers.map((header, index) => (
                                        <th key={`${header}-${index}`}>{header || `Coluna ${index + 1}`}</th>
                                    ))}
                                    <th>{csv.resultHeader || 'Resultado'}</th>
                                </tr>
                            </thead>
                            <tbody>
                                {previewRows.map((row, rowIndex) => (
                                    <tr key={rowIndex}>
                                        {csv.headers.map((_, cellIndex) => (
                                            <td key={cellIndex}>{row[cellIndex] ?? ''}</td>
                                        ))}
                                        <td className="result-cell">{csv.results[rowIndex] ?? ''}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </>
            ) : (
                <div className="empty">Carregue um CSV para visualizar as primeiras linhas.</div>
            )}
        </section>
    );
}

function Layer({ title, labels }) {
    return (
        <div className="layer">
            <p>{title}</p>
            <div className="nodes">
                {labels.map((label) => (
                    <span className="node" key={label}>
                        {label}
                    </span>
                ))}
            </div>
        </div>
    );
}

function NetworkGraph({ model, inputLabels, hiddenCount, outputLabels }) {
    const hiddenLabels = Array.from({ length: hiddenCount }, (_, index) => `H${index + 1}`);
    const limitedInputLabels = inputLabels.slice(0, 8);
    const limitedHiddenLabels = hiddenLabels.slice(0, 8);
    const limitedOutputLabels = outputLabels.slice(0, 8);
    const hasWeights = model.inputToHidden.length > 0 || model.hiddenToOutput.length > 0;

    return (
        <div className="network">
            <Layer title="Entrada" labels={formatLayerLabels(limitedInputLabels, inputLabels.length)} />
            <ConnectionBand
                title="Entrada -> Oculta"
                connections={model.inputToHidden}
                emptyLabel={hasWeights ? 'Sem pesos de entrada' : 'Gere ou treine pesos'}
            />
            <Layer title="Camada escondida" labels={formatLayerLabels(limitedHiddenLabels, hiddenLabels.length)} />
            <ConnectionBand
                title="Oculta -> Saida"
                connections={model.hiddenToOutput}
                emptyLabel={hasWeights ? 'Sem pesos de saida' : 'Gere ou treine pesos'}
            />
            <Layer title="Saida" labels={formatLayerLabels(limitedOutputLabels, outputLabels.length)} />
        </div>
    );
}

function formatLayerLabels(labels, totalCount) {
    if (totalCount <= labels.length) {
        return labels;
    }

    return [...labels, '...'];
}

function ConnectionBand({ title, connections, emptyLabel }) {
    const visibleConnections = connections.slice(0, 18);

    return (
        <div className="connection-band" aria-label={title}>
            <span>{title}</span>
            <div className="connection-lines">
                {visibleConnections.length > 0 ? (
                    visibleConnections.map((connection, index) => (
                        <button
                            className="connection-line"
                            key={`${connection.from}-${connection.to}-${index}`}
                            style={{
                                '--line-weight': `${Math.max(1, Math.min(5, Math.abs(connection.weight) * 5))}px`,
                            }}
                            title={`${connection.from} -> ${connection.to}: ${connection.weight.toFixed(6)}`}
                            type="button"
                        >
                            <span>{connection.weight.toFixed(4)}</span>
                        </button>
                    ))
                ) : (
                    <small>{emptyLabel}</small>
                )}
            </div>
        </div>
    );
}

function ModelPreview({ model, previousModel, weights }) {
    const safeModel = model ?? { inputToHidden: [], hiddenToOutput: [] };
    const inputToHiddenPreview = safeModel.inputToHidden.slice(0, 12);
    const hiddenToOutputPreview = safeModel.hiddenToOutput.slice(0, 12);
    const previousInputToHidden = previousModel?.inputToHidden ?? [];
    const previousHiddenToOutput = previousModel?.hiddenToOutput ?? [];

    if (weights.length === 0) {
        return <div className="model-preview empty-preview">Nenhum teste aleatório gerado ainda.</div>;
    }

    return (
        <div className="model-preview">
            <div className="model-stat">
                <span>Total de pesos</span>
                <strong>{weights.length}</strong>
            </div>
            <WeightList title="Entrada → Oculta" items={inputToHiddenPreview} previousItems={previousInputToHidden} />
            <WeightList title="Oculta → Saída" items={hiddenToOutputPreview} previousItems={previousHiddenToOutput} />
        </div>
    );
}

function WeightList({ title, items, previousItems }) {
    return (
        <div className="weight-list">
            <span>{title}</span>
            {items.length > 0 ? (
                <ul>
                    {items.map((item, index) => {
                        const previousWeight = previousItems[index]?.weight;
                        const changed = previousWeight !== undefined && previousWeight !== item.weight;

                        return (
                            <li className={changed ? 'weight-changed' : ''} key={`${item.from}-${item.to}`}>
                                <span>{item.from} → {item.to}</span>
                                <strong>{item.weight.toFixed(6)}</strong>
                                {changed ? <small>{previousWeight.toFixed(6)}</small> : null}
                            </li>
                        );
                    })}
                </ul>
            ) : (
                <p>Sem conexões geradas.</p>
            )}
        </div>
    );
}

function PredictTestPanel({ predictTest, setPredictTest, avgError }) {
    return (
        <div className="predict-panel">
            <div className="field compact-field">
                <span>Linha para testar predict</span>
                <input
                    type="number"
                    min="0"
                    value={predictTest.lineIndex}
                    onChange={(event) =>
                        setPredictTest((current) => ({
                            ...current,
                            lineIndex: event.target.value,
                        }))
                    }
                />
            </div>
            <div className="predict-summary">
                <span>Classe esperada</span>
                <strong>{predictTest.expectedResult || '-'}</strong>
            </div>
            <div className="predict-summary">
                <span>Classe obtida</span>
                <strong>{predictTest.predictedResult || '-'}</strong>
            </div>
            <div className={predictTest.correct === null ? 'predict-summary' : predictTest.correct ? 'predict-summary predict-hit' : 'predict-summary predict-miss'}>
                <span>Status</span>
                <strong>{predictTest.correct === null ? '-' : predictTest.correct ? 'Acerto' : 'Erro'}</strong>
            </div>
            <div className="predict-summary">
                <span>Erro medio treino</span>
                <strong>{Number(avgError || 0).toFixed(6)}</strong>
            </div>
            {predictTest.error ? <div className="error">{predictTest.error}</div> : null}
        </div>
    );
}

function TrainingPanel({ uniqueResults, trainingState, onContinueTraining, onReduceLearningRate }) {
    const lastHistory = trainingState.history.slice(-10).reverse();
    return (
        <section className="panel placeholder">
            <SectionTitle title="Treinamento" subtitle="Estado visual da preparação" />
            <div className="training-state">
                <span>Classes únicas armazenadas</span>
                {uniqueResults.length > 0 ? (
                    <div className="result-tags training-tags">
                        {uniqueResults.map((result) => (
                            <span key={result}>{result}</span>
                        ))}
                    </div>
                ) : (
                    <p>Nenhuma classe armazenada ainda.</p>
                )}
            </div>
            <div className="training-state">
                <span>Resultado do treinamento</span>
                <div className="training-metrics">
                    <strong>{trainingState.epochs || 0}</strong>
                    <small>epocas</small>
                    <strong>{Number(trainingState.avgError || 0).toFixed(6)}</strong>
                    <small>erro medio</small>
                    <strong>{trainingState.stoppedBy || '-'}</strong>
                    <small>parada</small>
                </div>
                {trainingState.plateau ? (
                    <>
                        <p>Possivel plato identificado nas ultimas 10 epocas.</p>
                        <div className="plateau-actions">
                            <button type="button" onClick={onContinueTraining}>
                                Continuar
                            </button>
                            <button type="button" onClick={onReduceLearningRate}>
                                Reduzir taxa 10%
                            </button>
                        </div>
                    </>
                ) : null}
            </div>
            {lastHistory.length > 0 ? (
                <ul className="history-list">
                    {lastHistory.map((item) => (
                        <li key={item.epoch}>
                            <span>Epoca {item.epoch}</span>
                            <strong>{item.avgError.toFixed(6)}</strong>
                        </li>
                    ))}
                </ul>
            ) : (
                <div className="empty">Execute o treinamento para ver o historico de epocas.</div>
            )}
        </section>
    );
}

function PlaceholderPanel({ title, lines }) {
    return (
        <section className="panel placeholder">
            <SectionTitle title={title} subtitle="Área reservada para etapa posterior" />
            <ul>
                {lines.map((line) => (
                    <li key={line}>{line}</li>
                ))}
            </ul>
        </section>
    );
}

function ConfusionMatrixPanel({ evaluationState }) {
    const hasMatrix = evaluationState.matrix.length > 0;

    return (
        <section className="panel">
            <SectionTitle title="Matriz de ConfusÃ£o" subtitle="Linhas: classe esperada. Colunas: classe obtida." />
            <div className="metrics">
                <span>{evaluationState.total || 0} exemplos</span>
                <span>{evaluationState.correct || 0} acertos</span>
                <span>{(evaluationState.accuracy * 100).toFixed(2)}% acuracia</span>
            </div>
            {hasMatrix ? (
                <div className="table-wrap confusion-wrap">
                    <table className="confusion-table">
                        <thead>
                            <tr>
                                <th>Esperada \ Obtida</th>
                                {evaluationState.labels.map((label) => (
                                    <th key={label}>{label}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {evaluationState.labels.map((expectedLabel, rowIndex) => (
                                <tr key={expectedLabel}>
                                    <th>{expectedLabel}</th>
                                    {evaluationState.labels.map((predictedLabel, columnIndex) => (
                                        <td
                                            className={rowIndex === columnIndex ? 'confusion-hit' : 'confusion-miss'}
                                            key={predictedLabel}
                                        >
                                            {evaluationState.matrix[rowIndex]?.[columnIndex] ?? 0}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            ) : (
                <div className="empty">Treine a rede e clique em Avaliar teste para gerar a matriz.</div>
            )}
        </section>
    );
}

function recomendationNumberHiddenNeurons({ numberOfInputs, numberOfOutputs }) {
    // Regra prática comum: utilizando da média aritimética chegamos em um número de neurônios na camada escondida entre a média e o máximo de neurônios nas camadas de entrada e saída
    const inputNeurons = numberOfInputs;
    const outputNeurons = numberOfOutputs;
    const hiddenNeurons = Math.floor((inputNeurons + outputNeurons) / 2);

    return hiddenNeurons;
}

function buildMlpWeights(inputCount, hiddenNeuronsCount, outputCount, weights, trainingUniqueResults) {
    if (weights.length !== inputCount * hiddenNeuronsCount + hiddenNeuronsCount * outputCount) {
        throw new Error('O número de pesos fornecidos não corresponde à arquitetura da rede.');
    }

    if (inputCount <= 0 || hiddenNeuronsCount <= 0 || outputCount <= 0) {
        throw new Error('O número de neurônios em cada camada deve ser maior que zero.');
    }

    if (weights.some((weight) => typeof weight !== 'number' || isNaN(weight) || weight.length === 0)) {
        throw new Error('Todos os pesos devem ser números válidos.');
    }
    let MLPWeights = {
        inputToHidden: [],
        hiddenToOutput: [],
    };

    for (let i = 0; i < inputCount; i += 1) {
        for (let j = 0; j < hiddenNeuronsCount; j += 1) {
            const weightIndex = i * hiddenNeuronsCount + j;
            const weightValue = weights[weightIndex];
            MLPWeights.inputToHidden.push({
                from: `Input ${i + 1}`,
                to: `Hidden ${j + 1}`,
                weight: weightValue,
            });
            console.log(`Peso da conexão entre entrada ${i} e neurônio oculto ${j}: ${weightValue}`);
        }
    }

    const hiddenToOutputStartIndex = inputCount * hiddenNeuronsCount;

    for (let i = 0; i < hiddenNeuronsCount; i += 1) {
        for (let j = 0; j < outputCount; j += 1) {
            const weightIndex = hiddenToOutputStartIndex + i * outputCount + j;
            const weightValue = weights[weightIndex];
            MLPWeights.hiddenToOutput.push({
                from: `Hidden ${i + 1}`,
                to: `Output ${j + 1}`,
                classe: trainingUniqueResults[j] || '',
                weight: weightValue,
            });
            console.log(`Peso da conexão entre neurônio oculto ${i} e saída ${j}: ${weightValue}`);
        }
    }

    const MlpInformation = {
        hidden: {
            net: [],
            erro: [],
            i: [],
        },
        output: {
            classe: [],
            net: [],
            erro: [],
            i: [],
        },
    };
    for (let i = 0; i < hiddenNeuronsCount; i += 1) {
        MlpInformation.hidden.net.push(0);
        MlpInformation.hidden.erro.push(0);
        MlpInformation.hidden.i.push(0);
    }

    for (let i = 0; i < outputCount; i += 1) {
        MlpInformation.output.classe.push(trainingUniqueResults[i] || '');
        MlpInformation.output.net.push(0);
        MlpInformation.output.erro.push(0);
        MlpInformation.output.i.push(0);
    }

    return { MLPWeights, MlpInformation };
}

function generateRandomWeights(inputCount, hiddenNeuronsCount, outputCount) {
    const inputToHiddenWeightsCount = inputCount * hiddenNeuronsCount;
    const hiddenToOutputWeightsCount = hiddenNeuronsCount * outputCount;
    const totalWeights = inputToHiddenWeightsCount + hiddenToOutputWeightsCount;

    const randomWeights = Array.from({ length: totalWeights }, () => Math.random());
    return randomWeights;
}

function getActivationFunction(name) {
    const normalizedName = String(name || '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();

    if (normalizedName.includes('logistica')) {
        return {
            run: (net) => 1 / (1 + Math.exp(-net)),
            derivative: (output) => output * (1 - output),
        };
    }

    if (normalizedName.includes('tangente')) {
        return {
            run: (net) => Math.tanh(net),
            derivative: (output) => 1 - output ** 2,
        };
    }

    return {
        run: (net) => net / 10,
        derivative: () => 1 / 10,
    };
}

function extractModelWeights(model) {
    return [
        ...model.inputToHidden.map((connection) => connection.weight),
        ...model.hiddenToOutput.map((connection) => connection.weight),
    ];
}

function standardDeviation(values) {
    if (values.length === 0) {
        return 0;
    }

    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;

    return Math.sqrt(variance);
}

function hasTrainingPlateau(history) {
    if (history.length < 10) {
        return false;
    }

    const lastErrors = history.slice(-10).map((item) => item.avgError);
    const deviation = standardDeviation(lastErrors);

    return deviation >= 0 && deviation <= 0.00001;
}

function cloneMlpModel(model) {
    return {
        inputToHidden: model.inputToHidden.map((connection) => ({ ...connection })),
        hiddenToOutput: model.hiddenToOutput.map((connection) => ({ ...connection })),
    };
}

function cloneLayerInformation(layer) {
    return Object.fromEntries(
        Object.entries(layer).map(([key, value]) => [key, Array.isArray(value) ? [...value] : value])
    );
}

function trainMlp({ rows, results, uniqueResults, hiddenNeuronsCount, maxEpochs, errorThreshold, config, initialState }) {
    const inputCount = rows[0]?.length ?? 0;
    const outputCount = uniqueResults.length;
    const shouldContinue = Boolean(initialState?.model?.inputToHidden?.length);
    const initialWeights = shouldContinue ? [] : generateRandomWeights(inputCount, hiddenNeuronsCount, outputCount);
    const builtMlp = shouldContinue ? null : buildMlpWeights(inputCount, hiddenNeuronsCount, outputCount, initialWeights, uniqueResults);

    let currentModel = shouldContinue ? cloneMlpModel(initialState.model) : builtMlp.MLPWeights;
    let currentHidden = shouldContinue ? cloneLayerInformation(initialState.hidden) : builtMlp.MlpInformation.hidden;
    let currentOutput = shouldContinue ? cloneLayerInformation(initialState.output) : builtMlp.MlpInformation.output;
    let avgError = Number.POSITIVE_INFINITY;
    let stoppedBy = 'maximo de epocas';
    const history = initialState?.history ? [...initialState.history] : [];
    const initialEpoch = history.at(-1)?.epoch ?? 0;

    for (let epoch = 1; epoch <= maxEpochs; epoch += 1) {
        let epochError = 0;

        for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
            const predictionResult = predict(currentHidden, currentOutput, currentModel, rows[rowIndex], results[rowIndex], config, 0);

            currentHidden = predictionResult.hidden;
            currentOutput = predictionResult.output;
            currentModel = predictionResult.model;
            epochError += predictionResult.erroRede;
        }

        avgError = epochError / rows.length;
        history.push({ epoch: initialEpoch + epoch, avgError });

        if (avgError <= errorThreshold) {
            stoppedBy = 'limiar de erro';
            break;
        }
    }

    return {
        model: currentModel,
        hidden: currentHidden,
        output: currentOutput,
        weights: extractModelWeights(currentModel),
        avgError,
        epochs: history.at(-1)?.epoch ?? 0,
        history,
        plateau: hasTrainingPlateau(history),
        stoppedBy,
    };
}

function runMlp(model, lineInputs, config) {
    const hiddenNeuronsCount = model.inputToHidden.reduce((max, connection) => {
        const hiddenIndex = Number.parseInt(connection.to.replace(/\D/g, ''), 10);
        return Math.max(max, hiddenIndex);
    }, 0);
    const outputLabels = [...new Set(model.hiddenToOutput.map((connection) => connection.classe).filter(Boolean))];
    const outputCount = outputLabels.length;
    const activation = getActivationFunction(config.activationFunction);
    const hiddenI = Array(hiddenNeuronsCount).fill(0);
    const outputI = Array(outputCount).fill(0);

    for (let hiddenIndex = 0; hiddenIndex < hiddenNeuronsCount; hiddenIndex += 1) {
        let net = 0;

        for (let inputIndex = 0; inputIndex < lineInputs.length; inputIndex += 1) {
            const weightIndex = inputIndex * hiddenNeuronsCount + hiddenIndex;
            const connection = model.inputToHidden[weightIndex];

            net += (connection?.weight ?? 0) * lineInputs[inputIndex];
        }

        hiddenI[hiddenIndex] = activation.run(net);
    }

    for (let outputIndex = 0; outputIndex < outputCount; outputIndex += 1) {
        let net = 0;

        for (let hiddenIndex = 0; hiddenIndex < hiddenNeuronsCount; hiddenIndex += 1) {
            const weightIndex = hiddenIndex * outputCount + outputIndex;
            const connection = model.hiddenToOutput[weightIndex];

            net += (connection?.weight ?? 0) * hiddenI[hiddenIndex];
        }

        outputI[outputIndex] = activation.run(net);
    }

    const bestOutputIndex = outputI.reduce((bestIndex, value, index) => (
        value > outputI[bestIndex] ? index : bestIndex
    ), 0);

    return {
        outputs: outputI,
        predictedLabel: outputLabels[bestOutputIndex] ?? '',
    };
}

function evaluateMlp({ rows, results, labels, model, config }) {
    const matrix = labels.map(() => labels.map(() => 0));
    const predictions = rows.map((row, rowIndex) => {
        const expectedLabel = results[rowIndex];
        const prediction = runMlp(model, row, config);
        const expectedIndex = labels.indexOf(expectedLabel);
        const predictedIndex = labels.indexOf(prediction.predictedLabel);

        if (expectedIndex >= 0 && predictedIndex >= 0) {
            matrix[expectedIndex][predictedIndex] += 1;
        }

        return {
            expectedLabel,
            predictedLabel: prediction.predictedLabel,
            outputs: prediction.outputs,
            correct: expectedLabel === prediction.predictedLabel,
        };
    });
    const correct = predictions.filter((prediction) => prediction.correct).length;
    const total = predictions.length;

    return {
        total,
        correct,
        accuracy: total > 0 ? correct / total : 0,
        labels,
        matrix,
        predictions,
    };
}

function predict(mlpHiddenInformation, mlpOutputInformation, mlpModel, lineInputs, expectedResult, config, avgError) {
    const hiddenNeuronsCount = mlpHiddenInformation.net.length;
    const inputCount = lineInputs.length;
    const outputCount = mlpOutputInformation.net.length;

    const hiddenNet = Array(hiddenNeuronsCount).fill(0);
    const hiddenI = Array(hiddenNeuronsCount).fill(0);
    const hiddenErro = Array(hiddenNeuronsCount).fill(0);

    const outputNet = Array(outputCount).fill(0);
    const outputI = Array(outputCount).fill(0);
    const outputErro = Array(outputCount).fill(0);
    const outputClasse = [...mlpOutputInformation.classe];

    const nextModel = {
        inputToHidden: mlpModel.inputToHidden.map((connection) => ({ ...connection })),
        hiddenToOutput: mlpModel.hiddenToOutput.map((connection) => ({ ...connection })),
    };

    const learningRate = Number(config.learningRate) || 0.1;
    const activation = getActivationFunction(config.activationFunction);

    // 1. Calcula NET e I da camada oculta.
    for (let hiddenIndex = 0; hiddenIndex < hiddenNeuronsCount; hiddenIndex += 1) {
        let net = 0;

        for (let inputIndex = 0; inputIndex < inputCount; inputIndex += 1) {
            const weightIndex = inputIndex * hiddenNeuronsCount + hiddenIndex;
            const connection = nextModel.inputToHidden[weightIndex];

            net += (connection?.weight ?? 0) * lineInputs[inputIndex];
        }

        hiddenNet[hiddenIndex] = net;
        hiddenI[hiddenIndex] = activation.run(net);
    }

    // 2. Calcula NET, I e erro da camada de saída.
    for (let outputIndex = 0; outputIndex < outputCount; outputIndex += 1) {
        let net = 0;

        for (let hiddenIndex = 0; hiddenIndex < hiddenNeuronsCount; hiddenIndex += 1) {
            const weightIndex = hiddenIndex * outputCount + outputIndex;
            const connection = nextModel.hiddenToOutput[weightIndex];

            net += (connection?.weight ?? 0) * hiddenI[hiddenIndex];
        }

        outputNet[outputIndex] = net;
        outputI[outputIndex] = activation.run(net);

        const desired = outputClasse[outputIndex] === expectedResult ? 1 : 0;
        outputErro[outputIndex] = (desired - outputI[outputIndex]) * activation.derivative(outputI[outputIndex]);
    }

    // 3. Salva os pesos antigos da camada oculta → saída.
    // O erro da camada oculta precisa usar os pesos ANTES da atualização.
    const hiddenToOutputWeightsBeforeUpdate =
        nextModel.hiddenToOutput.map((connection) => connection.weight);

    // 4. Calcula o erro da camada oculta.
    for (let hiddenIndex = 0; hiddenIndex < hiddenNeuronsCount; hiddenIndex += 1) {
        let sum = 0;

        for (let outputIndex = 0; outputIndex < outputCount; outputIndex += 1) {
            const weightIndex = hiddenIndex * outputCount + outputIndex;

            sum +=
                outputErro[outputIndex] *
                hiddenToOutputWeightsBeforeUpdate[weightIndex];
        }

        hiddenErro[hiddenIndex] = sum * activation.derivative(hiddenI[hiddenIndex]);
    }

    // 5. Atualiza os pesos da camada oculta → saída.
    for (let outputIndex = 0; outputIndex < outputCount; outputIndex += 1) {
        for (let hiddenIndex = 0; hiddenIndex < hiddenNeuronsCount; hiddenIndex += 1) {
            const weightIndex = hiddenIndex * outputCount + outputIndex;
            const connection = nextModel.hiddenToOutput[weightIndex];

            const newWeight =
                connection.weight +
                learningRate * outputErro[outputIndex] * hiddenI[hiddenIndex];

            nextModel.hiddenToOutput[weightIndex].weight = newWeight;
        }
    }

    // 6. Atualiza os pesos da camada entrada → oculta.
    for (let hiddenIndex = 0; hiddenIndex < hiddenNeuronsCount; hiddenIndex += 1) {
        for (let inputIndex = 0; inputIndex < inputCount; inputIndex += 1) {
            const weightIndex = inputIndex * hiddenNeuronsCount + hiddenIndex;
            const connection = nextModel.inputToHidden[weightIndex];

            const newWeight =
                connection.weight +
                learningRate * hiddenErro[hiddenIndex] * lineInputs[inputIndex];

            nextModel.inputToHidden[weightIndex].weight = newWeight;
        }
    }

    // 7. Calcula o erro da rede usando as saídas obtidas.
    let erroRede = 0;

    for (let outputIndex = 0; outputIndex < outputCount; outputIndex += 1) {
        const desired = outputClasse[outputIndex] === expectedResult ? 1 : 0;
        erroRede += Math.pow(desired - outputI[outputIndex], 2);
    }

    erroRede *= 0.5;

    const nextAvgError = avgError > 0 ? (erroRede + avgError) / 2 : erroRede;

    return {
        hidden: {
            ...mlpHiddenInformation,
            net: hiddenNet,
            erro: hiddenErro,
            i: hiddenI,
        },
        output: {
            ...mlpOutputInformation,
            net: outputNet,
            i: outputI,
            erro: outputErro,
        },
        model: nextModel,
        erroRede,
        avgError: nextAvgError,
    };
}

createRoot(document.getElementById('root')).render(<App />);
