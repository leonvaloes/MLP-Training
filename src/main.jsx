import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const emptyCsv = {
    name: '',
    headers: [],
    rows: [],
    results: [],
    resultHeader: '',
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
    const rows = parsedRows.map((row) => row.slice(0, -1));
    const results = parsedRows.map((row) => row.at(-1) ?? '');

    return { headers, rows, results, resultHeader };
}

function getUniqueResults(results) {
    return [...new Set(results.filter((result) => result !== undefined && result !== ''))];
}

function App() {
    const [mode, setMode] = useState('separate');
    const [trainCsv, setTrainCsv] = useState(emptyCsv);
    const [testCsv, setTestCsv] = useState(emptyCsv);
    const [singleCsv, setSingleCsv] = useState(emptyCsv);
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
    const [weights, setWeights] = useState([]);
    const [notification, setNotification] = useState(null);
    const [isNotificationHovered, setIsNotificationHovered] = useState(false);
    const activeCsv = mode === 'single' ? singleCsv : trainCsv;
    const testRows = mode === 'single' ? 0 : testCsv.rows.length;
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
            setter({
                name: file.name,
                headers: parsed.headers,
                rows: parsed.rows,
                results: parsed.results,
                resultHeader: parsed.resultHeader,
                error: '',
            });
            if (shouldStoreTrainingResults) {
                const uniqueTrainingResults = getUniqueResults(parsed.results);
                setTrainingUniqueResults(uniqueTrainingResults);
                setNumberOfOutputs(uniqueTrainingResults.length);
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

    function handlePredictTest() {
        if (weights.length === 0 || mlpModel.inputToHidden.length === 0) {
            setNotification({
                title: 'Modelo necessário',
                description: 'Gere pesos aleatórios antes de testar o predict.',
            });
            return;
        }

        const lineInputs = Array.from({ length: inputCount || 3 }, () => Number(Math.random().toFixed(4)));
        const predictionResult = predict(mlpHiddenInformation, mlpOutputInformation, mlpModel, lineInputs);

        setMlpHiddenInformation(predictionResult.hidden);
        setMlpOutputInformation(predictionResult.output);
        console.log('Teste predict:', {
            lineInputs,
            MlpModel,
            result: predictionResult,
        });
        setNotification({
            title: 'Predict testado',
            description: 'O resultado do predict foi enviado para o console.',
            details: [
                `${lineInputs.length} entradas usadas`,
                `${predictionResult.hidden.net.length} nets ocultos`,
                `${predictionResult.output.net.length} nets de saída`,
            ],
        });
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
                            <FileInput label="CSV completo" csv={singleCsv} onFile={(file) => readCsvFile(file, setSingleCsv, true)} />
                            <div className="notice">A divisão em treino/teste não foi implementada.</div>
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
                        <CsvPanel title="Base de Teste" csv={mode === 'single' ? emptyCsv : testCsv} />
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
                        </div>
                        <div className="network">
                            <Layer title="Entrada" labels={['X1', 'X2', 'X3', '...']} />
                            <div className="connector" />
                            <Layer title="Camada escondida" labels={['H1', 'H2', 'H3']} />
                            <div className="connector" />
                            <Layer title="Saída" labels={['C1', 'C2', 'C3', '...']} />
                        </div>
                        <ModelPreview model={mlpModel} weights={weights} />
                    </section>

                    <div className="grid two">
                        <TrainingPanel uniqueResults={trainingUniqueResults} />
                        <PlaceholderPanel
                            title="Matriz de Confusão"
                            lines={['Classes esperadas', 'Classes obtidas', 'Acertos na diagonal', 'Erros fora da diagonal']}
                        />
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

function ModelPreview({ model, weights }) {
    const safeModel = model ?? { inputToHidden: [], hiddenToOutput: [] };
    const inputToHiddenPreview = safeModel.inputToHidden.slice(0, 4);
    const hiddenToOutputPreview = safeModel.hiddenToOutput.slice(0, 4);

    if (weights.length === 0) {
        return <div className="model-preview empty-preview">Nenhum teste aleatório gerado ainda.</div>;
    }

    return (
        <div className="model-preview">
            <div className="model-stat">
                <span>Total de pesos</span>
                <strong>{weights.length}</strong>
            </div>
            <WeightList title="Entrada → Oculta" items={inputToHiddenPreview} />
            <WeightList title="Oculta → Saída" items={hiddenToOutputPreview} />
        </div>
    );
}

function WeightList({ title, items }) {
    return (
        <div className="weight-list">
            <span>{title}</span>
            {items.length > 0 ? (
                <ul>
                    {items.map((item) => (
                        <li key={`${item.from}-${item.to}`}>
                            {item.from} → {item.to}: <strong>{item.weight.toFixed(4)}</strong>
                        </li>
                    ))}
                </ul>
            ) : (
                <p>Sem conexões geradas.</p>
            )}
        </div>
    );
}

function TrainingPanel({ uniqueResults }) {
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
            <ul>
                {['Histórico de épocas', 'Média de erro', 'Estado de platô', 'Ações de continuidade'].map((line) => (
                    <li key={line}>{line}</li>
                ))}
            </ul>
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

function predict(mlpHiddenInformation, mlpOutputInformation, mlpModel, lineInputs, config, avgError) {
    const hiddenNeuronsCount = mlpHiddenInformation.net.length;
    const inputCount = lineInputs.length;
    const outputCount = mlpOutputInformation.net.length;
    const hiddenNet = [...mlpHiddenInformation.net];
    const hiddenI = [...mlpHiddenInformation.i];

    const outputNet = [...mlpOutputInformation.net];
    const outputI = [...mlpOutputInformation.i];
    const outputErro = [...mlpOutputInformation.erro];
    const outputClasse = [...mlpOutputInformation.classe];

    const classeValue = lineInputs.split(',').pop().trim();
    // Cálculo do net da camada de entrada para a camada oculta.
    for (let i = 0; i < hiddenNeuronsCount; i++) {
        let net = 0;
        for (let j = 0; j < inputCount; j++) {
            const indice = j * (hiddenNeuronsCount - 1) + i;
            const inputConnection = mlpModel.inputToHidden[indice];
            net += (inputConnection?.weight ?? 0) * lineInputs[j];
        }
        hiddenNet[i] = net;
        hiddenI[i] = net / 2;
    }

    // Cálculo do net da camada oculta para a camada de saída.
    for (let i = 0; i < outputCount; i++) {
        let net = 0;
        for (let j = 0; j < hiddenNeuronsCount; j++) {
            const indice = j * (outputCount - 1) + i;
            const hiddenConnection = mlpModel.hiddenToOutput[indice];
            net += (hiddenConnection?.weight ?? 0) * hiddenNet[j];
        }
        outputNet[i] = net;
        outputI[i] = net / 2;

        //Calculo do erro
        let auxI = 1 - net / 2;
        if (outputClasse[i] == classeValue) {
            outputErro[i] = (1 - auxI) * 0.5;
        } else {
            outputErro[i] = auxI * 0.5;
        }
    }

    //atualizar os pesos da camada ocuta para a camada de saída utilizando o erro calculado
    for (let i = 0; i < outputCount; i++) {
        for (let j = 0; j < hiddenNeuronsCount; j++) {
            const indice = j * (outputCount - 1) + i;
            const hiddenConnection = mlpModel.hiddenToOutput[indice];

            const newWeight = hiddenConnection.weight + config.learningRate * outputErro[i] * hiddenI[j];
            mlpModel.hiddenToOutput[indice].weight = newWeight;
        }
    }

    //atualizar os pesos da camada entrada para a camada oculta utilizando o erro calculado
    for (let i = 0; i < hiddenNeuronsCount; i++) {
        for (let j = 0; j < inputCount; j++) {
            const indice = j * (hiddenNeuronsCount - 1) + i;
            const inputConnection = mlpModel.inputToHidden[indice];
            const newWeight = inputConnection.weight + config.learningRate * outputErro[i];
        }
    }

    //erro da rede
    let erroRede = 0;
    for (let i = 0; i < outputCount; i++) {
        auxI = outputErro[i] / 2;
        erroRede += Math.pow(auxI, 2);
    }
    erroRede *= 0.5;

    avgError = (erroRede + avgError) / 2;
}

createRoot(document.getElementById('root')).render(<App />);
