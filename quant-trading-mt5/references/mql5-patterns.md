# MQL5 Implementation Patterns

## Table of Contents
1. [EA Structure](#ea-structure)
2. [Tick-Based Processing](#tick-based-processing)
3. [Order Management](#order-management)
4. [Position Sizing](#position-sizing)
5. [Risk Controls](#risk-controls)
6. [Indicator Integration](#indicator-integration)
7. [Session Filters](#session-filters)
8. [Error Handling](#error-handling)

---

## EA Structure

### Scalping EA Template

```mql5
#property copyright "Quant Trading"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\AccountInfo.mqh>

// Risk Management Inputs
input group "=== Risk Management ==="
input double RiskPerTrade = 1.0;           // Risk per trade (%)
input double MaxDailyDrawdown = 3.0;       // Max daily drawdown (%)
input double MaxTotalDrawdown = 10.0;      // Max total drawdown (%)
input int    MaxPositions = 3;             // Max concurrent positions
input int    MaxSpreadPoints = 20;         // Max allowed spread (points)

// Strategy Inputs
input group "=== Strategy Parameters ==="
input int    FastPeriod = 5;               // Fast EMA period
input int    SlowPeriod = 20;              // Slow EMA period
input double ZScoreThreshold = 2.0;        // Z-Score entry threshold
input int    LookbackPeriod = 100;         // Statistical lookback

// Session Inputs
input group "=== Session Filter ==="
input bool   UseLondonSession = true;      // Trade London session
input bool   UseNYSession = true;          // Trade NY session
input int    LondonStart = 8;              // London start (hour UTC)
input int    LondonEnd = 16;               // London end (hour UTC)
input int    NYStart = 13;                 // NY start (hour UTC)
input int    NYEnd = 21;                   // NY end (hour UTC)

// Objects
CTrade trade;
CPositionInfo positionInfo;
CAccountInfo accountInfo;

// State variables
double dailyStartBalance;
datetime lastTradeDay;
int totalTradesToday;
double dailyPnL;

//+------------------------------------------------------------------+
int OnInit() {
    // Validate inputs
    if(RiskPerTrade <= 0 || RiskPerTrade > 5) {
        Print("Invalid RiskPerTrade. Must be 0-5%");
        return INIT_PARAMETERS_INCORRECT;
    }

    // Initialize trade settings
    trade.SetExpertMagicNumber(123456);
    trade.SetDeviationInPoints(10);
    trade.SetTypeFilling(ORDER_FILLING_IOC);

    // Initialize daily tracking
    dailyStartBalance = accountInfo.Balance();
    lastTradeDay = TimeCurrent();
    totalTradesToday = 0;
    dailyPnL = 0;

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    // Cleanup
}

//+------------------------------------------------------------------+
void OnTick() {
    // Daily reset check
    CheckDailyReset();

    // Pre-trade checks
    if(!IsTradeAllowed()) return;
    if(!CheckSpread()) return;
    if(!CheckSession()) return;
    if(!CheckDrawdownLimits()) return;
    if(!CheckMaxPositions()) return;

    // Process tick for signals
    ProcessTickSignal();

    // Manage open positions
    ManagePositions();
}
```

---

## Tick-Based Processing

### Direct Tick Access

```mql5
void ProcessTickSignal() {
    MqlTick tick;
    if(!SymbolInfoTick(_Symbol, tick)) return;

    double bid = tick.bid;
    double ask = tick.ask;
    double spread = ask - bid;
    long tickVolume = tick.volume;

    // Tick-level analysis
    static double lastBid = 0;
    static double lastAsk = 0;
    static long tickCount = 0;

    if(lastBid > 0) {
        double bidChange = bid - lastBid;
        double askChange = ask - lastAsk;

        // Detect aggressive buying/selling
        if(bidChange > 0 && askChange > 0) {
            // Bid and Ask both rising - buying pressure
            OnBuyPressure(bidChange, tickVolume);
        }
        else if(bidChange < 0 && askChange < 0) {
            // Bid and Ask both falling - selling pressure
            OnSellPressure(MathAbs(bidChange), tickVolume);
        }
    }

    lastBid = bid;
    lastAsk = ask;
    tickCount++;
}
```

### Tick Aggregation for Analysis

```mql5
// Circular buffer for tick data
#define TICK_BUFFER_SIZE 1000

struct TickData {
    double bid;
    double ask;
    long volume;
    datetime time;
    long timeMsc;
};

TickData tickBuffer[];
int tickIndex = 0;
int tickCount = 0;

void InitTickBuffer() {
    ArrayResize(tickBuffer, TICK_BUFFER_SIZE);
}

void AddTick(MqlTick &tick) {
    tickBuffer[tickIndex].bid = tick.bid;
    tickBuffer[tickIndex].ask = tick.ask;
    tickBuffer[tickIndex].volume = tick.volume;
    tickBuffer[tickIndex].time = tick.time;
    tickBuffer[tickIndex].timeMsc = tick.time_msc;

    tickIndex = (tickIndex + 1) % TICK_BUFFER_SIZE;
    if(tickCount < TICK_BUFFER_SIZE) tickCount++;
}

double CalculateTickZScore(int period) {
    if(tickCount < period) return 0;

    // Get recent price changes
    double changes[];
    ArrayResize(changes, period - 1);

    int idx = tickIndex - 1;
    for(int i = 0; i < period - 1; i++) {
        int curr = (idx - i + TICK_BUFFER_SIZE) % TICK_BUFFER_SIZE;
        int prev = (curr - 1 + TICK_BUFFER_SIZE) % TICK_BUFFER_SIZE;
        changes[i] = tickBuffer[curr].bid - tickBuffer[prev].bid;
    }

    // Calculate z-score of most recent change
    double mean = 0, stdDev = 0;
    for(int i = 1; i < period - 1; i++) mean += changes[i];
    mean /= (period - 2);

    for(int i = 1; i < period - 1; i++)
        stdDev += MathPow(changes[i] - mean, 2);
    stdDev = MathSqrt(stdDev / (period - 2));

    if(stdDev == 0) return 0;
    return (changes[0] - mean) / stdDev;
}
```

---

## Order Management

### Reliable Order Execution

```mql5
bool OpenBuyPosition(double lots, double slPrice, double tpPrice, string comment = "") {
    MqlTradeRequest request = {};
    MqlTradeResult result = {};

    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = NormalizeDouble(lots, 2);
    request.type = ORDER_TYPE_BUY;
    request.price = NormalizeDouble(ask, digits);
    request.sl = NormalizeDouble(slPrice, digits);
    request.tp = NormalizeDouble(tpPrice, digits);
    request.deviation = 10;
    request.magic = trade.RequestMagic();
    request.comment = comment;
    request.type_filling = ORDER_FILLING_IOC;

    // Retry logic
    for(int attempt = 0; attempt < 3; attempt++) {
        ResetLastError();

        if(OrderSend(request, result)) {
            if(result.retcode == TRADE_RETCODE_DONE ||
               result.retcode == TRADE_RETCODE_PLACED) {
                Print("Buy order placed: ", result.order, " at ", result.price);
                return true;
            }
        }

        // Handle requotes
        if(result.retcode == TRADE_RETCODE_REQUOTE) {
            request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            Sleep(100);
            continue;
        }

        Print("Order failed: ", result.retcode, " - ", GetRetcodeDescription(result.retcode));
        break;
    }

    return false;
}

bool OpenSellPosition(double lots, double slPrice, double tpPrice, string comment = "") {
    MqlTradeRequest request = {};
    MqlTradeResult result = {};

    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = NormalizeDouble(lots, 2);
    request.type = ORDER_TYPE_SELL;
    request.price = NormalizeDouble(bid, digits);
    request.sl = NormalizeDouble(slPrice, digits);
    request.tp = NormalizeDouble(tpPrice, digits);
    request.deviation = 10;
    request.magic = trade.RequestMagic();
    request.comment = comment;
    request.type_filling = ORDER_FILLING_IOC;

    for(int attempt = 0; attempt < 3; attempt++) {
        ResetLastError();

        if(OrderSend(request, result)) {
            if(result.retcode == TRADE_RETCODE_DONE ||
               result.retcode == TRADE_RETCODE_PLACED) {
                Print("Sell order placed: ", result.order, " at ", result.price);
                return true;
            }
        }

        if(result.retcode == TRADE_RETCODE_REQUOTE) {
            request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            Sleep(100);
            continue;
        }

        Print("Order failed: ", result.retcode, " - ", GetRetcodeDescription(result.retcode));
        break;
    }

    return false;
}

string GetRetcodeDescription(uint retcode) {
    switch(retcode) {
        case TRADE_RETCODE_REQUOTE: return "Requote";
        case TRADE_RETCODE_REJECT: return "Request rejected";
        case TRADE_RETCODE_CANCEL: return "Request canceled";
        case TRADE_RETCODE_PLACED: return "Order placed";
        case TRADE_RETCODE_DONE: return "Request completed";
        case TRADE_RETCODE_INVALID: return "Invalid request";
        case TRADE_RETCODE_INVALID_VOLUME: return "Invalid volume";
        case TRADE_RETCODE_INVALID_PRICE: return "Invalid price";
        case TRADE_RETCODE_INVALID_STOPS: return "Invalid stops";
        case TRADE_RETCODE_NO_MONEY: return "Insufficient funds";
        default: return "Unknown error";
    }
}
```

### Position Modification

```mql5
bool ModifyStopLoss(ulong ticket, double newSL) {
    if(!PositionSelectByTicket(ticket)) return false;

    MqlTradeRequest request = {};
    MqlTradeResult result = {};

    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

    request.action = TRADE_ACTION_SLTP;
    request.position = ticket;
    request.symbol = _Symbol;
    request.sl = NormalizeDouble(newSL, digits);
    request.tp = PositionGetDouble(POSITION_TP);

    return OrderSend(request, result) &&
           (result.retcode == TRADE_RETCODE_DONE);
}

bool ClosePosition(ulong ticket) {
    if(!PositionSelectByTicket(ticket)) return false;

    MqlTradeRequest request = {};
    MqlTradeResult result = {};

    double volume = PositionGetDouble(POSITION_VOLUME);
    ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

    request.action = TRADE_ACTION_DEAL;
    request.position = ticket;
    request.symbol = _Symbol;
    request.volume = volume;
    request.deviation = 10;
    request.type_filling = ORDER_FILLING_IOC;

    if(posType == POSITION_TYPE_BUY) {
        request.type = ORDER_TYPE_SELL;
        request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    } else {
        request.type = ORDER_TYPE_BUY;
        request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    }

    return OrderSend(request, result) &&
           (result.retcode == TRADE_RETCODE_DONE);
}
```

---

## Position Sizing

### Fixed Percentage Risk

```mql5
double CalculateLotSize(double riskPercent, double slPoints) {
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = balance * riskPercent / 100.0;

    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    double pointValue = tickValue / tickSize;

    double lots = riskAmount / (slPoints * pointValue);

    // Normalize to lot step
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    lots = MathFloor(lots / lotStep) * lotStep;
    lots = MathMax(minLot, MathMin(maxLot, lots));

    return NormalizeDouble(lots, 2);
}
```

### Kelly Criterion

```mql5
double CalculateKellyLots(double winRate, double avgWin, double avgLoss, double fraction = 0.25) {
    // Kelly formula: f* = (bp - q) / b
    // where b = win/loss ratio, p = win probability, q = loss probability

    if(avgLoss == 0 || winRate <= 0 || winRate >= 1) return 0;

    double b = MathAbs(avgWin / avgLoss);  // Win/loss ratio
    double p = winRate;
    double q = 1 - winRate;

    double kelly = (b * p - q) / b;

    // Use fractional Kelly (safer)
    kelly *= fraction;

    if(kelly <= 0) return 0;

    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = balance * kelly;

    // Convert to lots using average loss
    double lots = riskAmount / MathAbs(avgLoss);

    // Normalize
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    lots = MathFloor(lots / lotStep) * lotStep;
    lots = MathMax(minLot, MathMin(maxLot, lots));

    return NormalizeDouble(lots, 2);
}
```

### ATR-Based Sizing

```mql5
double CalculateATRLots(double riskPercent, double atrMultiplier = 2.0) {
    int atrHandle = iATR(_Symbol, PERIOD_H1, 14);
    if(atrHandle == INVALID_HANDLE) return 0;

    double atrBuffer[];
    ArraySetAsSeries(atrBuffer, true);

    if(CopyBuffer(atrHandle, 0, 0, 1, atrBuffer) <= 0) return 0;

    double atr = atrBuffer[0];
    double slPoints = atr * atrMultiplier / _Point;

    IndicatorRelease(atrHandle);

    return CalculateLotSize(riskPercent, slPoints);
}
```

---

## Risk Controls

### Drawdown Circuit Breaker

```mql5
bool CheckDrawdownLimits() {
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);

    // Daily drawdown check
    double dailyDD = (dailyStartBalance - equity) / dailyStartBalance * 100;
    if(dailyDD >= MaxDailyDrawdown) {
        Print("Daily drawdown limit reached: ", DoubleToString(dailyDD, 2), "%");
        CloseAllPositions();
        return false;
    }

    // Total drawdown check
    static double peakBalance = 0;
    if(balance > peakBalance) peakBalance = balance;

    double totalDD = (peakBalance - equity) / peakBalance * 100;
    if(totalDD >= MaxTotalDrawdown) {
        Print("Total drawdown limit reached: ", DoubleToString(totalDD, 2), "%");
        CloseAllPositions();
        return false;
    }

    return true;
}

void CheckDailyReset() {
    MqlDateTime now;
    TimeToStruct(TimeCurrent(), now);

    MqlDateTime lastTrade;
    TimeToStruct(lastTradeDay, lastTrade);

    if(now.day != lastTrade.day) {
        dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
        lastTradeDay = TimeCurrent();
        totalTradesToday = 0;
        dailyPnL = 0;
        Print("New trading day. Balance: ", dailyStartBalance);
    }
}
```

### Spread Filter

```mql5
bool CheckSpread() {
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double spread = (ask - bid) / _Point;

    if(spread > MaxSpreadPoints) {
        // Print("Spread too high: ", spread, " points");
        return false;
    }

    return true;
}
```

### Position Limit

```mql5
bool CheckMaxPositions() {
    int openPositions = 0;
    int magicNumber = (int)trade.RequestMagic();

    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        if(positionInfo.SelectByIndex(i)) {
            if(positionInfo.Symbol() == _Symbol &&
               positionInfo.Magic() == magicNumber) {
                openPositions++;
            }
        }
    }

    return openPositions < MaxPositions;
}

void CloseAllPositions() {
    int magicNumber = (int)trade.RequestMagic();

    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        if(positionInfo.SelectByIndex(i)) {
            if(positionInfo.Symbol() == _Symbol &&
               positionInfo.Magic() == magicNumber) {
                ClosePosition(positionInfo.Ticket());
            }
        }
    }
}
```

---

## Session Filters

```mql5
bool CheckSession() {
    if(!UseLondonSession && !UseNYSession) return true;

    MqlDateTime now;
    TimeToStruct(TimeGMT(), now);
    int hour = now.hour;

    bool inLondon = UseLondonSession && (hour >= LondonStart && hour < LondonEnd);
    bool inNY = UseNYSession && (hour >= NYStart && hour < NYEnd);

    return inLondon || inNY;
}

bool IsHighImpactNewsTime() {
    // Check if within 30 minutes of known news events
    // Implement with calendar or external data feed

    // Placeholder - implement based on economic calendar
    return false;
}
```

---

## Error Handling

```mql5
void LogError(string function, int errorCode) {
    string errorDesc = ErrorDescription(errorCode);
    Print("Error in ", function, ": [", errorCode, "] ", errorDesc);
}

string ErrorDescription(int errorCode) {
    switch(errorCode) {
        case ERR_SUCCESS: return "Success";
        case ERR_NO_MQLERROR: return "No error";
        case ERR_TRADE_DISABLED: return "Trading disabled";
        case ERR_MARKET_CLOSED: return "Market closed";
        case ERR_NOT_ENOUGH_MONEY: return "Insufficient funds";
        case ERR_TRADE_NOT_ALLOWED: return "Trading not allowed";
        case ERR_INVALID_STOPS: return "Invalid stops";
        default: return "Unknown error: " + IntegerToString(errorCode);
    }
}

bool IsTradeAllowed() {
    if(!MQLInfoInteger(MQL_TRADE_ALLOWED)) {
        Print("Trading not allowed in EA settings");
        return false;
    }

    if(!AccountInfoInteger(ACCOUNT_TRADE_ALLOWED)) {
        Print("Trading not allowed on account");
        return false;
    }

    if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)) {
        Print("Trading not allowed in terminal");
        return false;
    }

    ENUM_SYMBOL_TRADE_MODE tradeMode =
        (ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE);

    if(tradeMode != SYMBOL_TRADE_MODE_FULL) {
        Print("Symbol trading mode restricted: ", EnumToString(tradeMode));
        return false;
    }

    return true;
}
```
