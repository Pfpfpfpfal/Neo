var network = new SimpleNeuralNetwork(3);

var layerFactory = new NeuralLayerFactory();
network.AddLayer(layerFactory.CreateNeuralLayer(3, new RectifiedActivationFuncion(), new WeightedSumFunction()));
network.AddLayer(layerFactory.CreateNeuralLayer(1, new SigmoidActivationFunction(0.7), new WeightedSumFunction()));

network.PushExpectedValues(
    new double[][] {
        new double[] { 0 },
        new double[] { 1 },
        new double[] { 1 },
        new double[] { 0 },
        new double[] { 1 },
        new double[] { 0 },
        new double[] { 0 },
    });

network.Train(
    new double[][] {
        new double[] { 150, 2, 0 },
        new double[] { 1002, 56, 1 },
        new double[] { 1060, 59, 1 },
        new double[] { 200, 3, 0 },
        new double[] { 300, 3, 1 },
        new double[] { 120, 1, 0 },
        new double[] { 80, 1, 0 },
    }, 10000);

network.PushInputValues(new double[] { 1054, 54, 1 });
var outputs = network.GetOutput();
public interface IInputFunction
{
    double CalculateInput(List<ISynapse> inputs);//CalculateInput, получает список подключений, описанных в  интерфейсе ISynapse
    //CalculateInput  должен возвращать какое-то значение на основе данных, содержащихся в списке подключений

}
public class WeightedSumFunction : IInputFunction//входная функция
{
    //суммирует взвешенные значения по всем соединениям, которые передаются в списке
    public double CalculateInput(List<ISynapse> inputs)
    {
        return inputs.Select(x => x.Weight * x.GetOutput()).Sum();
    }
}
public interface IActivationFunction//функция активации
{
    double CalculateOutput(double input);
    //CalculateOutput  должен возвращать выходное значение нейрона на основе входного значения, полученного от входной функции.
}
public class StepActivationFunction : IActivationFunction//ступенчатая функция
{
    //построения объекта
    //CalculateOutput возвращает 1, если входное значение превышает пороговое значение, иначе возвращает 0.
    private double _treshold;

    public StepActivationFunction(double treshold)
    {
        _treshold = treshold;
    }

    public double CalculateOutput(double input)
    {
        return Convert.ToDouble(input > _treshold);
    }
}
public class SigmoidActivationFunction : IActivationFunction//активация сигмоида
{
    private double _coeficient;

    public SigmoidActivationFunction(double coeficient)
    {
        _coeficient = coeficient;
    }

    public double CalculateOutput(double input)
    {
        return (1 / (1 + Math.Exp(-input * _coeficient)));
    }
}
public class RectifiedActivationFuncion : IActivationFunction//активация выпрямителя
{
    public double CalculateOutput(double input)
    {
        return Math.Max(0, input);
    }
}
public interface INeuron//нейрон
{
    //получение входных значений от одного или нескольких взвешенных входных соединений
    //сбор значений и передача их функции активации, которая вычисляет выходное значение нейрона
    //отправление значений на выходы нейрона
    Guid Id { get; }
    double PreviousPartialDerivate { get; set; }

    List<ISynapse> Inputs { get; set; }
    List<ISynapse> Outputs { get; set; }

    void AddInputNeuron(INeuron inputNeuron);
    void AddOutputNeuron(INeuron inputNeuron);
    double CalculateOutput();

    void AddInputSynapse(double inputValue);
    void PushValueOnInput(double inputValue);
}
public class Neuron : INeuron//реализация нейрона
{
    private IActivationFunction _activationFunction;
    private IInputFunction _inputFunction;

    //Входные связи нейрона
    public List<ISynapse> Inputs { get; set; }

    //Выходные связи нейрона
    public List<ISynapse> Outputs { get; set; }

    public Guid Id { get; private set; }

    //Вычислить частную производную в предыдущей итерации процесса обучения
    public double PreviousPartialDerivate { get; set; }

    public Neuron(IActivationFunction activationFunction, IInputFunction inputFunction)
    {
        Id = Guid.NewGuid();
        Inputs = new List<ISynapse>();
        Outputs = new List<ISynapse>();

        _activationFunction = activationFunction;
        _inputFunction = inputFunction;
    }

    // Соединение двух нейронов.
    // Этот нейрон является выходным нейроном соединения
    // 
    // inputNeuron - нейрон, который будет входным нейроном вновь созданного соединения
    public void AddInputNeuron(INeuron inputNeuron)
    {
        var synapse = new Synapse(inputNeuron, this);
        Inputs.Add(synapse);
        inputNeuron.Outputs.Add(synapse);
    }

    // Соединение двух нейронов.
    // Этот нейрон является входным нейроном соединения
    // outputNeuron - нейрон, который будет выходным нейроном вновь созданного соединения
    public void AddOutputNeuron(INeuron outputNeuron)
    {
        var synapse = new Synapse(this, outputNeuron);
        Outputs.Add(synapse);
        outputNeuron.Inputs.Add(synapse);
    }
    // Рассчитать выходное значение нейрона
    // Выход нейрона
    public double CalculateOutput()
    {
        return _activationFunction.CalculateOutput(_inputFunction.CalculateInput(this.Inputs));
    }


    // Нейроны входного слоя получают входные значения
    // AddInputSynapse - добавляет соединение к нейрону

    // inputValue
    // Начальное значение, которое будет добавлено в качестве входных данных для подключения
    public void AddInputSynapse(double inputValue)
    {
        var inputSynapse = new InputSynapse(this, inputValue);
        Inputs.Add(inputSynapse);
    }

    // Устанавливает новое значение для входных соединений

    // inputValue
    // Новое значение, которое будет добавлено в качестве входных данных для подключения
    public void PushValueOnInput(double inputValue)
    {
        ((InputSynapse)Inputs.First()).Output = inputValue;
    }
}
public interface ISynapse//соединения
{
    //Каждое соединение имеет свой вес
    double Weight { get; set; }
    double PreviousWeight { get; set; }//используется при обратном распространении ошибки системой
    double GetOutput();

    bool IsFromNeuron(Guid fromNeuronId);//определяет, является ли данный нейрон входным нейроном
    void UpdateWeight(double learningRate, double delta);//Обновление текущего веса и сохранение предыдущего
}

public class Synapse : ISynapse//подключение
{
    /// <summary>
    /// _fromNeuron и _toNeuron-определяют нейроны, которые соединяет этот синапс
    /// </summary>
    internal INeuron _fromNeuron;
    internal INeuron _toNeuron;

    // Вес соединения
    public double Weight { get; set; }

    // Прошлый вес
    public double PreviousWeight { get; set; }

    public Synapse(INeuron fromNeuraon, INeuron toNeuron, double weight)
    {
        _fromNeuron = fromNeuraon;
        _toNeuron = toNeuron;

        Weight = weight;
        PreviousWeight = 0;
    }

    public Synapse(INeuron fromNeuraon, INeuron toNeuron)
    {
        _fromNeuron = fromNeuraon;
        _toNeuron = toNeuron;

        var tmpRandom = new Random();
        Weight = tmpRandom.NextDouble();
        PreviousWeight = 0;
    }

    // Получить выходное значение соединения
    // Выходное значение соединения
    public double GetOutput()
    {
        return _fromNeuron.CalculateOutput();
    }

    // имеет ли Нейрон определенный номер в качестве входного нейрона
    // 
    // fromNeuronId" Neuron Id
    // 
    // True - если нейрон является входом связи
    // False - iесли нейрон не является входом связи 

    public bool IsFromNeuron(Guid fromNeuronId)
    {
        return _fromNeuron.Id.Equals(fromNeuronId);
    }

    // Обновление веса
    //
    // learningRate-скорость обчучения
    // delta-Расчетная разница, для которой необходимо изменить вес соединения
    public void UpdateWeight(double learningRate, double delta)
    {
        PreviousWeight = Weight;
        Weight += learningRate * delta;
    }
}
public class InputSynapse : ISynapse//используется как вход в систему
{
    internal INeuron _toNeuron;

    public double Weight { get; set; }
    public double Output { get; set; }
    public double PreviousWeight { get; set; }

    public InputSynapse(INeuron toNeuron)
    {
        _toNeuron = toNeuron;
        Weight = 1;
    }

    public InputSynapse(INeuron toNeuron, double output)
    {
        _toNeuron = toNeuron;
        Output = output;
        Weight = 1;
        PreviousWeight = 1;
    }

    public double GetOutput()
    {
        return Output;
    }

    public bool IsFromNeuron(Guid fromNeuronId)
    {
        return false;
    }

    public void UpdateWeight(double learningRate, double delta)
    {
        throw new InvalidOperationException("It is not allowed to call this method on Input Connecion");
    }
}
public class NeuralLayer//нейронный слой
{
    public List<INeuron> Neurons;

    public NeuralLayer()
    {
        Neurons = new List<INeuron>();
    }

    /// <summary>
    /// Вставка двух слоев
    /// </summary>
    public void ConnectLayers(NeuralLayer inputLayer)
    {
        var combos = Neurons.SelectMany(neuron => inputLayer.Neurons, (neuron, input) => new { neuron, input });
        combos.ToList().ForEach(x => x.neuron.AddInputNeuron(x.input));
    }
}
public class SimpleNeuralNetwork//нейроная сеть
{
    //содержит список нейронных слоев и фабрику слоев, класс, который используется для создания новых слоев
    private NeuralLayerFactory _layerFactory;

    internal List<NeuralLayer> _layers;
    internal double _learningRate;
    internal double[][] _expectedResult;

    /// <summary>
    /// Конструктор сети
    /// </summary>
    /// numberOfInputNeurons-
    /// Количество нейронов во входном слое
    public SimpleNeuralNetwork(int numberOfInputNeurons)
    {
        _layers = new List<NeuralLayer>();
        _layerFactory = new NeuralLayerFactory();

        // Create input layer that will collect inputs.
        CreateInputLayer(numberOfInputNeurons);

        _learningRate = 2.95;
    }

    /// <summary>
    /// Добавление слоя
    /// Будет автоматически добавлен в качестве выходного слоя к последнему слою в сети
    /// </summary>
    public void AddLayer(NeuralLayer newLayer)
    {
        if (_layers.Any())
        {
            var lastLayer = _layers.Last();
            newLayer.ConnectLayers(lastLayer);
        }

        _layers.Add(newLayer);
    }

    /// <summary>
    /// Отправка входных значений в сеть
    /// </summary>
    public void PushInputValues(double[] inputs)
    {
        _layers.First().Neurons.ForEach(x => x.PushValueOnInput(inputs[_layers.First().Neurons.IndexOf(x)]));
    }

    /// <summary>
    /// Установка ожидаемых значений для выходных данных
    /// </summary>
    public void PushExpectedValues(double[][] expectedOutputs)
    {
        _expectedResult = expectedOutputs;
    }

    /// <summary>
    /// Рассчет выхода нейронной сети
    /// </summary>
    public List<double> GetOutput()
    {
        var returnValue = new List<double>();

        _layers.Last().Neurons.ForEach(neuron =>
        {
            returnValue.Add(neuron.CalculateOutput());
        });

        return returnValue;
    }

    /// <summary>
    /// Тренировка сети
    /// </summary>
    /// inputs - Входные значения
    /// numberOfEpochs-Количество эпох
    public void Train(double[][] inputs, int numberOfEpochs)
    {
        double totalError = 0;

        for (int i = 0; i < numberOfEpochs; i++)
        {
            for (int j = 0; j < inputs.GetLength(0); j++)
            {
                PushInputValues(inputs[j]);

                var outputs = new List<double>();

                _layers.Last().Neurons.ForEach(x => //Получить результаты
                {
                    outputs.Add(x.CalculateOutput());
                });

                // вычислить ошибку путем суммирования ошибок на всех выходных нейронах
                totalError = CalculateTotalError(outputs, j);
                HandleOutputLayer(j);
                HandleHiddenLayers();
            }
        }
    }

    /// <summary>
    /// создает входной слой сети
    /// </summary>
    private void CreateInputLayer(int numberOfInputNeurons)
    {
        var inputLayer = _layerFactory.CreateNeuralLayer(numberOfInputNeurons, new RectifiedActivationFuncion(), new WeightedSumFunction());
        inputLayer.Neurons.ForEach(x => x.AddInputSynapse(0));
        this.AddLayer(inputLayer);
    }

    /// <summary>
    /// вычисляет общую ошибку сети
    /// </summary>
    private double CalculateTotalError(List<double> outputs, int row)
    {
        double totalError = 0;

        outputs.ForEach(output =>
        {
            var error = Math.Pow(output - _expectedResult[row][outputs.IndexOf(output)], 2);
            totalError += error;
        });

        return totalError;
    }

    /// <summary>
    /// запускает алгоритм обратного распространения на выходном слое сети
    /// </summary>
    /// row-
    /// входная/выходная ожидаемая строка
    private void HandleOutputLayer(int row)
    {
        _layers.Last().Neurons.ForEach(neuron =>
        {
            neuron.Inputs.ForEach(connection =>
            {
                var output = neuron.CalculateOutput();
                var netInput = connection.GetOutput();

                var expectedOutput = _expectedResult[row][_layers.Last().Neurons.IndexOf(neuron)];

                var nodeDelta = (expectedOutput - output) * output * (1 - output);
                var delta = -1 * netInput * nodeDelta;

                connection.UpdateWeight(_learningRate, delta);

                neuron.PreviousPartialDerivate = nodeDelta;
            });
        });
    }

    /// <summary>
    /// запускает алгоритм обратного распространения на скрытом слое сети
    /// </summary>
    /// row-
    /// входная/выходная ожидаемая строка
    private void HandleHiddenLayers()
    {
        for (int k = _layers.Count - 2; k > 0; k--)
        {
            _layers[k].Neurons.ForEach(neuron =>
            {
                neuron.Inputs.ForEach(connection =>
                {
                    var output = neuron.CalculateOutput();
                    var netInput = connection.GetOutput();
                    double sumPartial = 0;

                    _layers[k + 1].Neurons
                    .ForEach(outputNeuron =>
                    {
                        outputNeuron.Inputs.Where(i => i.IsFromNeuron(neuron.Id))
                        .ToList()
                        .ForEach(outConnection =>
                        {
                            sumPartial += outConnection.PreviousWeight * outputNeuron.PreviousPartialDerivate;
                        });
                    });

                    var delta = -1 * netInput * sumPartial * output * (1 - output);
                    connection.UpdateWeight(_learningRate, delta);
                });
            });
        }
    }
}