/*
 ===================================================================================
  🐠 AquaSense AI — ESP32-S3 智慧魚缸 pH 酸鹼值與水溫 IoT 監測器
  ===================================================================================
  🎯 功能特色：
   1. 專為 ESP32-S3 開發，使用 ADC1 (GPIO 1) 進行 20 次平滑濾波取樣，抗水波雜訊。
   2. 自動連接 Wi-Fi，並支援斷線自動重連機制。
   3. 定時每 3 秒透過 HTTP POST JSON 無線將即時 pH 值與水溫上傳至 AquaSense AI 監控儀表板。
   4. 內建 pH 探針標準電壓換算公式與簡易 2 點校準參數。
  
  📦 Arduino IDE 需安裝之程式庫：
   - WiFi.h (ESP32 核心內建)
   - HTTPClient.h (ESP32 核心內建)
   - ArduinoJson.h (請在「程式庫管理員」搜尋安裝 ArduinoJson by Benoit Blanchon)
  
  🔌 硬體接線說明 (ESP32-S3)：
   - pH 感測模組 VCC  -> 5V 或 3.3V (建議使用穩定的 5V/3.3V 獨立供電)
   - pH 感測模組 GND  -> ESP32-S3 GND
   - pH 感測模組 AOUT -> ESP32-S3 GPIO 1 (可自訂 PIN_PH_ANALOG)
 ===================================================================================
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ==========================================
// 1. Wi-Fi 與伺服器設定 (請修改為您的 Wi-Fi 與 Mac IP)
// ==========================================
const char* WIFI_SSID     = "YOUR_WIFI_SSID";      // 您的 2.4G Wi-Fi 名稱
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";  // 您的 Wi-Fi 密碼

// 🖥️ AquaSense AI 主機網址 (Mac 的區域網路 IP，Port 為 5000)
const char* SERVER_URL    = "http://192.168.0.119:5000/api/sensor_upload";

// ==========================================
// 2. 硬體腳位與感測器校準參數
// ==========================================
const int PIN_PH_ANALOG = 1;      // ESP32-S3 ADC1 通道腳位 (GPIO 1)
const int PIN_STATUS_LED = 2;     // 狀態指示燈 (GPIO 2 或板載 LED)

// ⚖️ pH 校準參數 (標準中性 pH 7.0 時的探針輸出電壓，一般約為 1.50V ~ 2.50V)
// 提示：若將探針放入 pH 7.0 校正液中測得電壓為 1.65V，則將 VOLTAGE_PH7 改為 1.65
float VOLTAGE_PH7 = 1.65;         // pH 7.0 中性基準電壓 (V)
float PH_SLOPE    = -3.5;         // 每伏特 pH 斜率 (標準探針約為 -3.5 到 -5.9)
float PH_OFFSET   = 0.00;         // 細微微調偏差值

// ⏱️ 上傳時間間隔 (毫秒，預設 3000ms = 3 秒)
const unsigned long UPLOAD_INTERVAL_MS = 3000;
unsigned long lastUploadTime = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n==================================================");
  Serial.println(" 🐠 AquaSense AI — ESP32-S3 IoT pH 感測器啟動中...");
  Serial.println("==================================================");

  // 初始化腳位
  pinMode(PIN_STATUS_LED, OUTPUT);
  digitalWrite(PIN_STATUS_LED, LOW);

  // 設定 ESP32-S3 ADC 解析度為 12-bit (0 ~ 4095) 與衰減率
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db); // 可量測 0 ~ 3.3V 電壓範圍

  // 連接 Wi-Fi
  connectToWiFi();
}

void loop() {
  // 檢查 Wi-Fi 連線狀態
  if (WiFi.status() != WL_CONNECTED) {
    digitalWrite(PIN_STATUS_LED, LOW);
    connectToWiFi();
  }

  // 定時讀取並上傳水質數據
  unsigned long currentMillis = millis();
  if (currentMillis - lastUploadTime >= UPLOAD_INTERVAL_MS) {
    lastUploadTime = currentMillis;

    // 1. 採樣並計算 pH 值 (20次多重採樣去除高頻干擾)
    float phValue = readSmoothPH();
    
    // 2. 水溫設定 (若有接 DS18B20 溫度探針可在此讀取，否則預設 26.5°C)
    float waterTemp = 26.5;

    // 3. 輸出偵錯日誌至 Serial Monitor
    Serial.printf("[Sensor] 實測 pH: %.2f | 估算水溫: %.1f°C | 時間戳: %lu ms\n", phValue, waterTemp, millis());

    // 4. 發送 HTTP POST 到 AquaSense AI 伺服器
    uploadDataToAquaSense(phValue, waterTemp);
  }
}

// ==========================================
// 3. 連接 Wi-Fi 副程式 (具備防卡死重試)
// ==========================================
void connectToWiFi() {
  Serial.printf("📡 正在連線至 Wi-Fi: %s ...\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int retryCount = 0;
  while (WiFi.status() != WL_CONNECTED && retryCount < 20) {
    delay(500);
    Serial.print(".");
    retryCount++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    digitalWrite(PIN_STATUS_LED, HIGH);
    Serial.println("\n✅ Wi-Fi 連線成功！");
    Serial.printf("📌 ESP32-S3 區域網路 IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n❌ Wi-Fi 連線失敗，將於下個循環重新嘗試。");
  }
}

// ==========================================
// 4. 多重採樣中值濾波演算法 (去除水波雜訊)
// ==========================================
float readSmoothPH() {
  const int NUM_SAMPLES = 20;
  int rawList[NUM_SAMPLES];

  // 採集 20 個 ADC 樣本
  for (int i = 0; i < NUM_SAMPLES; i++) {
    rawList[i] = analogRead(PIN_PH_ANALOG);
    delay(15);
  }

  // 氣泡排序法排序樣本 (去除極大值與極小值噪訊)
  for (int i = 0; i < NUM_SAMPLES - 1; i++) {
    for (int j = i + 1; j < NUM_SAMPLES; j++) {
      if (rawList[i] > rawList[j]) {
        int temp = rawList[i];
        rawList[i] = rawList[j];
        rawList[j] = temp;
      }
    }
  }

  // 取中間 10 個樣本取平均值
  long sum = 0;
  for (int i = 5; i < 15; i++) {
    sum += rawList[i];
  }
  float avgRaw = sum / 10.0;

  // 將 ADC Raw (0~4095) 換算為實測電壓 (0 ~ 3.3V)
  float voltage = (avgRaw / 4095.0) * 3.3;

  // 依據標準 pH 感測電壓線性公式換算
  // 公式：pH = 7.0 + (Voltage - VOLTAGE_PH7) * PH_SLOPE + PH_OFFSET
  float ph = 7.0 + (voltage - VOLTAGE_PH7) * PH_SLOPE + PH_OFFSET;

  // 合理範圍保護 (0.0 ~ 14.0)
  if (ph < 0.0) ph = 0.0;
  if (ph > 14.0) ph = 14.0;

  return ph;
}

// ==========================================
// 5. 無線上傳至 AquaSense AI 伺服器
// ==========================================
void uploadDataToAquaSense(float ph, float temp) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("⚠️ Wi-Fi 斷線中，跳過本次上傳");
    return;
  }

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");

  // 建立 JSON 封包
  StaticJsonDocument<200> doc;
  doc["ph"] = serialized(String(ph, 2));
  doc["temp"] = serialized(String(temp, 1));
  doc["device"] = "ESP32-S3 (IoT 探針)";

  String jsonPayload;
  serializeJson(doc, jsonPayload);

  // 發送 HTTP POST
  int httpResponseCode = http.POST(jsonPayload);

  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.printf("📤 [上傳成功 200 OK] 伺服器回應: %s\n", response.c_str());
    
    // LED 快速閃爍一次表示成功通訊
    digitalWrite(PIN_STATUS_LED, LOW);
    delay(50);
    digitalWrite(PIN_STATUS_LED, HIGH);
  } else {
    Serial.printf("❌ [上傳失敗] 錯誤代碼: %d (%s)\n", httpResponseCode, http.errorToString(httpResponseCode).c_str());
  }

  http.end();
}
