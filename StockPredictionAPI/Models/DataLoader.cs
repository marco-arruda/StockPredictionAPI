using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace StockPredictionAPI.Models
{
    public class DataLoader
    {
        public static IEnumerable<StockData> LoadStockData(string dataPath)
        {
            var mlContext = new MLContext();

            // Carrega dados do arquivo
            var dataView = mlContext.Data.LoadFromTextFile<StockData>(dataPath, separatorChar: ',', hasHeader: true);
            return mlContext.Data.CreateEnumerable<StockData>(dataView, reuseRowObject: false).ToList();
        }
    }
}