using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
// using Azure.Storage.Blobs;


namespace MTN.Function
{
    public static class HttpTriggerTeamEmpireState
    {
        [FunctionName("HttpTriggerTeamEmpireState")]
        public static async Task<IActionResult> Run(
            [HttpTrigger("get", Route = "")] HttpRequest req,
            ILogger log)
        {
            string jsonResponse = @"{""2020-05-22 00:00:00"": 142, ""2020-05-22 01:00:00"": 160, " + 
            @"""2020-05-22 02:00:00"": 141, ""2020-05-22 03:00:00"": 161, ""2020-05-22 04:00:00"": 150, " +
            @"""2020-05-22 05:00:00"": 157, ""2020-05-22 06:00:00"": 148, ""2020-05-22 07:00:00"": 144, " +
            @"""2020-05-22 08:00:00"": 155, ""2020-05-22 09:00:00"": 136, ""2020-05-22 10:00:00"": 160, " + 
            @"""2020-05-22 11:00:00"": 147, ""2020-05-22 12:00:00"": 150, ""2020-05-22 13:00:00"": 141, " +
            @"""2020-05-22 14:00:00"": 142, ""2020-05-22 15:00:00"": 163, ""2020-05-22 16:00:00"": 148, " +
            @"""2020-05-22 17:00:00"": 157, ""2020-05-22 18:00:00"": 143, ""2020-05-22 19:00:00"": 160, " +
            @"""2020-05-22 20:00:00"": 155, ""2020-05-22 21:00:00"": 146, ""2020-05-22 22:00:00"": 147, " +
            @"""2020-05-22 23:00:00"": 158, ""2020-05-23 00:00:00"": 164, " +
            @"""2020-05-23 01:00:00"": 144, ""2020-05-23 02:00:00"": 152, ""2020-05-23 03:00:00"": 244, " +
            @"""2020-05-23 04:00:00"": 152, ""2020-05-23 05:00:00"": 157, ""2020-05-23 06:00:00"": 159, " +
            @"""2020-05-23 07:00:00"": 149, ""2020-05-23 08:00:00"": 139, ""2020-05-23 09:00:00"": 142, " +
            @"""2020-05-23 10:00:00"": 211, ""2020-05-23 11:00:00"": 154, ""2020-05-23 12:00:00"": 151, " +
            @"""2020-05-23 13:00:00"": 167, ""2020-05-23 14:00:00"": 155, ""2020-05-23 15:00:00"": 143, " +
            @"""2020-05-23 16:00:00"": 139, ""2020-05-23 17:00:00"": 191, ""2020-05-23 18:00:00"": 158, ""2020-05-23 19:00:00"": 159, " + 
            @"""2020-05-23 20:00:00"": 156, ""2020-05-23 21:00:00"": 164, ""2020-05-23 22:00:00"": 151, " + 
            @"""2020-05-23 23:00:00"": 137, ""2020-05-24 00:00:00"": 184, ""2020-05-24 01:00:00"": 159, " + 
            @"""2020-05-24 02:00:00"": 145, ""2020-05-24 03:00:00"": 156, ""2020-05-24 04:00:00"": 168, " + 
            @"""2020-05-24 05:00:00"": 0, ""2020-05-24 06:00:00"": 0, ""2020-05-24 07:00:00"": 0, " + 
            @"""2020-05-24 08:00:00"": 0, ""2020-05-24 09:00:00"": 0, ""2020-05-24 10:00:00"": 53, ""2020-05-24 11:00:00"": 56, " + 
            @"""2020-05-24 12:00:00"": 60, ""2020-05-24 13:00:00"": 67, ""2020-05-24 14:00:00"": 73, " + 
            @"""2020-05-24 15:00:00"": 62, ""2020-05-24 16:00:00"": 65, ""2020-05-24 17:00:00"": 56, " + 
            @"""2020-05-24 18:00:00"": 53, ""2020-05-24 19:00:00"": 54, ""2020-05-24 20:00:00"": 57, " + 
            @"""2020-05-24 21:00:00"": 58, ""2020-05-24 22:00:00"": 52, ""2020-05-24 23:00:00"": 60, " + 
            @"""2020-05-25 00:00:00"": 59, ""2020-05-25 01:00:00"": 61, ""2020-05-25 02:00:00"": 56, " + 
            @"""2020-05-25 03:00:00"": 112, ""2020-05-25 04:00:00"": 99, ""2020-05-25 05:00:00"": 106, " + 
            @"""2020-05-25 06:00:00"": 100, ""2020-05-25 07:00:00"": 93, ""2020-05-25 08:00:00"": 99, " + 
            @"""2020-05-25 09:00:00"": 115, ""2020-05-25 10:00:00"": 85, ""2020-05-25 11:00:00"": 76, " + 
            @"""2020-05-25 12:00:00"": 78, ""2020-05-25 13:00:00"": 85, ""2020-05-25 14:00:00"": 79, " + 
            @"""2020-05-25 15:00:00"": 85, ""2020-05-25 16:00:00"": 92, ""2020-05-25 17:00:00"": 108, " + 
            @"""2020-05-25 18:00:00"": 69, ""2020-05-25 19:00:00"": 74, ""2020-05-25 20:00:00"": 71, " + 
            @"""2020-05-25 21:00:00"": 71, ""2020-05-25 22:00:00"": 76, ""2020-05-25 23:00:00"": 81, " + 
            @"""2020-05-26 00:00:00"": 84, ""2020-05-26 01:00:00"": 62, ""2020-05-26 02:00:00"": 72, " + 
            @"""2020-05-26 03:00:00"": 65, ""2020-05-26 04:00:00"": 71, ""2020-05-26 05:00:00"": 71, " + 
            @"""2020-05-26 06:00:00"": 72, ""2020-05-26 07:00:00"": 77, ""2020-05-26 08:00:00"": 65, " + 
            @"""2020-05-26 09:00:00"": 72, ""2020-05-26 10:00:00"": 93, ""2020-05-26 11:00:00"": 107, " + 
            @"""2020-05-26 12:00:00"": 100, ""2020-05-26 13:00:00"": 108, ""2020-05-26 14:00:00"": 108, " + 
            @"""2020-05-26 15:00:00"": 100, ""2020-05-26 16:00:00"": 115, ""2020-05-26 17:00:00"": 112, " + 
            @"""2020-05-26 18:00:00"": 115, ""2020-05-26 19:00:00"": 124, ""2020-05-26 20:00:00"": 135, " + 
            @"""2020-05-26 21:00:00"": 127, ""2020-05-26 22:00:00"": 118, ""2020-05-26 23:00:00"": 122, " + 
            @"""2020-05-27 00:00:00"": 128, ""2020-05-27 01:00:00"": 123, ""2020-05-27 02:00:00"": 95, " + 
            @"""2020-05-27 03:00:00"": 98, ""2020-05-27 04:00:00"": 100, ""2020-05-27 05:00:00"": 97, " + 
            @"""2020-05-27 06:00:00"": 97, ""2020-05-27 07:00:00"": 95, ""2020-05-27 08:00:00"": 97, " + 
            @"""2020-05-27 09:00:00"": 78, ""2020-05-27 10:00:00"": 83, " + 
            @"""2020-05-27 11:00:00"": 81, ""2020-05-27 12:00:00"": 84, ""2020-05-27 13:00:00"": 80, " + 
            @"""2020-05-27 14:00:00"": 86, ""2020-05-27 15:00:00"": 78, ""2020-05-27 16:00:00"": 71, " + 
            @"""2020-05-27 17:00:00"": 107, ""2020-05-27 18:00:00"": 113, ""2020-05-27 19:00:00"": 125, " + 
            @"""2020-05-27 20:00:00"": 119, ""2020-05-27 21:00:00"": 125, ""2020-05-27 22:00:00"": 109, " + 
            @"""2020-05-27 23:00:00"": 109, ""2020-05-28 00:00:00"": 121, ""2020-05-28 01:00:00"": 137, " + 
            @"""2020-05-28 02:00:00"": 145, ""2020-05-28 03:00:00"": 123, ""2020-05-28 04:00:00"": 134, " + 
            @"""2020-05-28 05:00:00"": 128, ""2020-05-28 06:00:00"": 133, ""2020-05-28 07:00:00"": 126, " + 
            @"""2020-05-28 08:00:00"": 152, ""2020-05-28 09:00:00"": 140, ""2020-05-28 10:00:00"": 146, " + 
            @"""2020-05-28 11:00:00"": 134, ""2020-05-28 12:00:00"": 139, ""2020-05-28 13:00:00"": 139," + 
            @" ""2020-05-28 14:00:00"": 138, ""2020-05-28 15:00:00"": 158, ""2020-05-28 16:00:00"": 140, " + 
            @"""2020-05-28 17:00:00"": 155, ""2020-05-28 18:00:00"": 138, ""2020-05-28 19:00:00"": 147, " + 
            @"""2020-05-28 20:00:00"": 137, ""2020-05-28 21:00:00"": 144, ""2020-05-28 22:00:00"": 148, ""2020-05-28 23:00:00"": 154}";

            return new OkObjectResult(jsonResponse);
        }
    }
}
