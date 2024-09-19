using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;
using System.Collections.Generic;
using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;

namespace StockPredictionAPI.Services
{
    public class StockPredictionModel
    {
        private readonly string _modelPath;
        private PredictionEngine<StockData, StockPrediction> _model; // Correção aqui

        public StockPredictionModel(string modelPath)
        {
            _modelPath = modelPath;
        }

        public void LoadModel()
        {
            var mlContext = new MLContext();

            using var session = new InferenceSession(_modelPath);
            var inputSchema = session.InputMetadata;

            // Corrigido: Acesse a chave do primeiro item no dicionário
            var inputColumnName = inputSchema.First().Key;

            // Crie o pipeline de predição
            var pipeline = mlContext.Transforms.ApplyOnnxModel(
                _modelPath,
                inputColumnNames: inputSchema.Select(x => x.Key).ToArray(),
                outputColumnNames: new[] { "Predictions" }
            );

            // Criar um estimador para conversão de tipo (se necessário)
            var typeConverter = mlContext.Transforms.Conversion.ConvertType(
                outputColumnName: inputColumnName, // Nome da coluna de saída
                inputColumnName: inputColumnName, // Nome da coluna de entrada (mesmo nome)
                type: Microsoft.ML.Data.DataKind.Double // Tipo de dados desejado
            );

            // Adicione a conversão de tipo ao pipeline (se necessário)
            pipeline = pipeline.Append(typeConverter);

            // Crie o PredictionEngine
            _model = mlContext.Model.CreatePredictionEngine<StockData, StockPrediction>(pipeline);
        }

        public List<float[]> Predict(IEnumerable<StockData> data) // Ajuste no retorno
        {
            if (_model == null)
            {
                throw new InvalidOperationException("O modelo TensorFlow não foi carregado.");
            }

            // Fazer a previsão para cada item em 'data'
            var predictions = data.Select(stockData =>
            {
                var prediction = _model.Predict(stockData);
                return new float[] {
                    prediction.PredictedValues[0], // Abertura
                    prediction.PredictedValues[1], // Máxima
                    prediction.PredictedValues[2], // Mínima
                    prediction.PredictedValues[3], // Volume
                    prediction.PredictedValues[4]  // Variação
                };
            }).ToList();

            return predictions; // Retorna uma lista de arrays de floats
        }

        public class StockPrediction
        {
            [ColumnName("Predictions")]
            public float[] PredictedValues { get; set; }
        }
    }
}