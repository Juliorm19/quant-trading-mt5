# Risk Management Framework

## Table of Contents
1. [Position Sizing Methods](#position-sizing-methods)
2. [Drawdown Controls](#drawdown-controls)
3. [Correlation Management](#correlation-management)
4. [Trailing Stops](#trailing-stops)
5. [Risk Budgeting](#risk-budgeting)
6. [Emergency Procedures](#emergency-procedures)

---

## Position Sizing Methods

### Fixed Percentage Risk

Most conservative and widely used method. Risks a fixed percentage of account per trade.

**Formula:**
```
Lots = (Balance × RiskPercent) / (SL_pips × PipValue)
```

**MQL5 Implementation:**
```mql5
double FixedPercentLots(double riskPercent, double slPips) {
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = balance * riskPercent / 100.0;

    double pipValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE) /
                      SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE) * _Point;

    double lots = riskAmount / (slPips * pipValue);

    return NormalizeLots(lots);
}

double NormalizeLots(double lots) {
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    lots = MathFloor(lots / lotStep) * lotStep;
    return MathMax(minLot, MathMin(maxLot, lots));
}
```

**Risk Scaling by Account Size:**
| Account Size | Risk Per Trade | Daily Risk Limit |
|--------------|----------------|------------------|
| < $10,000 | 1.0% | 3% |
| $10k-$50k | 0.75% | 2.5% |
| $50k-$100k | 0.5% | 2% |
| > $100k | 0.25-0.5% | 1.5% |

---

### Kelly Criterion

Optimal position sizing for maximum long-term growth. Use fractional Kelly (25-50%) to reduce variance.

**Formula:**
```
f* = (bp - q) / b

where:
- b = average win / average loss (reward-to-risk ratio)
- p = probability of winning
- q = probability of losing (1 - p)
```

**MQL5 Implementation:**
```mql5
class CKellyCalculator {
private:
    double wins[];
    double losses[];
    int maxHistory;

public:
    CKellyCalculator(int historySize = 100) {
        maxHistory = historySize;
    }

    void AddTrade(double pnl) {
        if(pnl > 0) {
            ArrayResize(wins, ArraySize(wins) + 1);
            wins[ArraySize(wins) - 1] = pnl;
            if(ArraySize(wins) > maxHistory)
                ArrayRemove(wins, 0, 1);
        } else if(pnl < 0) {
            ArrayResize(losses, ArraySize(losses) + 1);
            losses[ArraySize(losses) - 1] = MathAbs(pnl);
            if(ArraySize(losses) > maxHistory)
                ArrayRemove(losses, 0, 1);
        }
    }

    double GetKellyFraction(double fraction = 0.25) {
        int totalTrades = ArraySize(wins) + ArraySize(losses);
        if(totalTrades < 30) return 0.01;  // Minimum sample

        double winRate = (double)ArraySize(wins) / totalTrades;
        double lossRate = 1 - winRate;

        double avgWin = 0, avgLoss = 0;
        for(int i = 0; i < ArraySize(wins); i++) avgWin += wins[i];
        for(int i = 0; i < ArraySize(losses); i++) avgLoss += losses[i];

        avgWin /= ArraySize(wins);
        avgLoss /= ArraySize(losses);

        if(avgLoss == 0) return 0.01;

        double b = avgWin / avgLoss;
        double kelly = (b * winRate - lossRate) / b;

        // Apply fraction and bounds
        kelly *= fraction;
        kelly = MathMax(0.005, MathMin(0.05, kelly));  // 0.5% to 5% bounds

        return kelly;
    }

    double GetLots(double slPips) {
        double kelly = GetKellyFraction();
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        double riskAmount = balance * kelly;

        double pipValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE) /
                          SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE) * _Point;

        double lots = riskAmount / (slPips * pipValue);
        return NormalizeLots(lots);
    }
};
```

**When to Use Kelly:**
- After 50+ trades with stable win rate
- When edge is well-quantified
- For accounts that can tolerate higher variance
- Always use fractional Kelly (25-50%)

---

### ATR-Based Position Sizing

Adjusts position size based on current market volatility. Larger positions in calm markets, smaller in volatile.

**Formula:**
```
SL = ATR × Multiplier
Lots = RiskAmount / (SL × PipValue)
```

**MQL5 Implementation:**
```mql5
class CATRPositionSizer {
private:
    int atrHandle;
    int atrPeriod;
    double atrMultiplier;
    double riskPercent;

public:
    CATRPositionSizer(int period = 14, double multiplier = 2.0, double risk = 1.0) {
        atrPeriod = period;
        atrMultiplier = multiplier;
        riskPercent = risk;
        atrHandle = INVALID_HANDLE;
    }

    bool Init(string symbol, ENUM_TIMEFRAMES tf = PERIOD_H1) {
        atrHandle = iATR(symbol, tf, atrPeriod);
        return atrHandle != INVALID_HANDLE;
    }

    void Deinit() {
        if(atrHandle != INVALID_HANDLE)
            IndicatorRelease(atrHandle);
    }

    double GetATR() {
        double buffer[];
        ArraySetAsSeries(buffer, true);
        if(CopyBuffer(atrHandle, 0, 0, 1, buffer) <= 0) return 0;
        return buffer[0];
    }

    double GetStopLoss(ENUM_ORDER_TYPE type) {
        double atr = GetATR();
        double slDistance = atr * atrMultiplier;

        if(type == ORDER_TYPE_BUY)
            return SymbolInfoDouble(_Symbol, SYMBOL_BID) - slDistance;
        else
            return SymbolInfoDouble(_Symbol, SYMBOL_ASK) + slDistance;
    }

    double GetLots() {
        double atr = GetATR();
        if(atr == 0) return 0;

        double slPips = (atr * atrMultiplier) / _Point;
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        double riskAmount = balance * riskPercent / 100.0;

        double pipValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE) /
                          SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE) * _Point;

        double lots = riskAmount / (slPips * pipValue);
        return NormalizeLots(lots);
    }
};
```

**ATR Multipliers by Strategy:**
| Strategy Type | ATR Multiplier | Typical SL |
|---------------|----------------|------------|
| Scalping | 0.5 - 1.0 | 5-15 pips |
| Intraday | 1.5 - 2.0 | 20-40 pips |
| Swing | 2.0 - 3.0 | 50-100 pips |

---

## Drawdown Controls

### Multi-Level Circuit Breakers

```mql5
enum ENUM_RISK_LEVEL {
    RISK_NORMAL,
    RISK_CAUTION,
    RISK_REDUCED,
    RISK_STOPPED
};

class CDrawdownController {
private:
    double peakBalance;
    double dailyStartBalance;
    datetime currentDay;

    // Thresholds
    double dailyCautionDD;    // Yellow alert
    double dailyMaxDD;        // Red - stop for day
    double totalCautionDD;
    double totalMaxDD;

public:
    CDrawdownController() {
        peakBalance = 0;
        dailyStartBalance = 0;
        currentDay = 0;
        dailyCautionDD = 2.0;
        dailyMaxDD = 3.0;
        totalCautionDD = 7.0;
        totalMaxDD = 10.0;
    }

    void Init(double cautionDaily = 2.0, double maxDaily = 3.0,
              double cautionTotal = 7.0, double maxTotal = 10.0) {
        dailyCautionDD = cautionDaily;
        dailyMaxDD = maxDaily;
        totalCautionDD = cautionTotal;
        totalMaxDD = maxTotal;
        Reset();
    }

    void Reset() {
        peakBalance = AccountInfoDouble(ACCOUNT_BALANCE);
        dailyStartBalance = peakBalance;
        currentDay = TimeCurrent();
    }

    void Update() {
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        if(balance > peakBalance) peakBalance = balance;

        MqlDateTime now, last;
        TimeToStruct(TimeCurrent(), now);
        TimeToStruct(currentDay, last);

        if(now.day != last.day) {
            dailyStartBalance = balance;
            currentDay = TimeCurrent();
        }
    }

    double GetDailyDrawdown() {
        double equity = AccountInfoDouble(ACCOUNT_EQUITY);
        return (dailyStartBalance - equity) / dailyStartBalance * 100;
    }

    double GetTotalDrawdown() {
        double equity = AccountInfoDouble(ACCOUNT_EQUITY);
        return (peakBalance - equity) / peakBalance * 100;
    }

    ENUM_RISK_LEVEL GetRiskLevel() {
        Update();

        double dailyDD = GetDailyDrawdown();
        double totalDD = GetTotalDrawdown();

        if(dailyDD >= dailyMaxDD || totalDD >= totalMaxDD)
            return RISK_STOPPED;

        if(dailyDD >= dailyCautionDD || totalDD >= totalCautionDD)
            return RISK_REDUCED;

        if(dailyDD >= dailyCautionDD * 0.7 || totalDD >= totalCautionDD * 0.7)
            return RISK_CAUTION;

        return RISK_NORMAL;
    }

    double GetRiskMultiplier() {
        switch(GetRiskLevel()) {
            case RISK_NORMAL:  return 1.0;
            case RISK_CAUTION: return 0.75;
            case RISK_REDUCED: return 0.5;
            case RISK_STOPPED: return 0;
        }
        return 0;
    }

    bool CanTrade() {
        return GetRiskLevel() != RISK_STOPPED;
    }
};
```

### Recovery Mode

After significant drawdown, reduce risk until recovery:

```mql5
class CRecoveryManager {
private:
    double drawdownThreshold;
    double recoveryMultiplier;
    bool inRecovery;
    double recoveryStartBalance;

public:
    CRecoveryManager(double threshold = 5.0, double multiplier = 0.5) {
        drawdownThreshold = threshold;
        recoveryMultiplier = multiplier;
        inRecovery = false;
        recoveryStartBalance = 0;
    }

    void Check(double currentDD) {
        if(!inRecovery && currentDD >= drawdownThreshold) {
            inRecovery = true;
            recoveryStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
            Print("Entering recovery mode at DD: ", DoubleToString(currentDD, 2), "%");
        }

        if(inRecovery) {
            double balance = AccountInfoDouble(ACCOUNT_BALANCE);
            // Exit recovery when we've recovered half the loss
            if(balance >= recoveryStartBalance * 1.025) {
                inRecovery = false;
                Print("Exiting recovery mode");
            }
        }
    }

    double GetRiskMultiplier() {
        return inRecovery ? recoveryMultiplier : 1.0;
    }

    bool IsInRecovery() {
        return inRecovery;
    }
};
```

---

## Correlation Management

Prevent over-exposure to correlated positions:

```mql5
class CCorrelationFilter {
private:
    string correlatedPairs[][2];
    int maxCorrelated;

public:
    CCorrelationFilter(int maxPositions = 2) {
        maxCorrelated = maxPositions;

        // Define correlated pairs
        // High positive correlation
        string pairs[5][2] = {
            {"EURUSD", "GBPUSD"},
            {"AUDUSD", "NZDUSD"},
            {"USDCHF", "USDJPY"},
            {"XAUUSD", "XAGUSD"},
            {"EURUSD", "EURJPY"}
        };

        ArrayResize(correlatedPairs, 5);
        for(int i = 0; i < 5; i++) {
            correlatedPairs[i][0] = pairs[i][0];
            correlatedPairs[i][1] = pairs[i][1];
        }
    }

    bool CanOpenPosition(string symbol, ENUM_ORDER_TYPE type) {
        // Count positions in correlated symbols with same direction
        int correlatedCount = 0;

        for(int i = PositionsTotal() - 1; i >= 0; i--) {
            string posSymbol = PositionGetSymbol(i);
            ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

            // Check if same direction
            bool sameDirection =
                (type == ORDER_TYPE_BUY && posType == POSITION_TYPE_BUY) ||
                (type == ORDER_TYPE_SELL && posType == POSITION_TYPE_SELL);

            if(!sameDirection) continue;

            // Check correlation
            if(IsCorrelated(symbol, posSymbol)) {
                correlatedCount++;
            }
        }

        return correlatedCount < maxCorrelated;
    }

    bool IsCorrelated(string symbol1, string symbol2) {
        if(symbol1 == symbol2) return true;

        for(int i = 0; i < ArrayRange(correlatedPairs, 0); i++) {
            if((correlatedPairs[i][0] == symbol1 && correlatedPairs[i][1] == symbol2) ||
               (correlatedPairs[i][0] == symbol2 && correlatedPairs[i][1] == symbol1)) {
                return true;
            }
        }
        return false;
    }
};
```

---

## Trailing Stops

### ATR-Based Trailing

```mql5
class CATRTrailingStop {
private:
    int atrHandle;
    double atrMultiplier;

public:
    CATRTrailingStop(string symbol, ENUM_TIMEFRAMES tf, int period, double mult) {
        atrMultiplier = mult;
        atrHandle = iATR(symbol, tf, period);
    }

    ~CATRTrailingStop() {
        if(atrHandle != INVALID_HANDLE)
            IndicatorRelease(atrHandle);
    }

    void Update(ulong ticket) {
        if(!PositionSelectByTicket(ticket)) return;

        double atr[];
        ArraySetAsSeries(atr, true);
        if(CopyBuffer(atrHandle, 0, 0, 1, atr) <= 0) return;

        double trailDistance = atr[0] * atrMultiplier;
        double currentSL = PositionGetDouble(POSITION_SL);
        double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
        int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

        ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

        if(posType == POSITION_TYPE_BUY) {
            double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            double newSL = NormalizeDouble(bid - trailDistance, digits);

            // Only move SL up, and only if in profit
            if(newSL > currentSL && newSL > openPrice) {
                ModifyStopLoss(ticket, newSL);
            }
        }
        else {
            double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            double newSL = NormalizeDouble(ask + trailDistance, digits);

            // Only move SL down, and only if in profit
            if((currentSL == 0 || newSL < currentSL) && newSL < openPrice) {
                ModifyStopLoss(ticket, newSL);
            }
        }
    }
};
```

### Breakeven Move

```mql5
void MoveToBreakeven(ulong ticket, double triggerPips, double lockPips = 1) {
    if(!PositionSelectByTicket(ticket)) return;

    double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
    double currentSL = PositionGetDouble(POSITION_SL);
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

    ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

    if(posType == POSITION_TYPE_BUY) {
        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double profit = (bid - openPrice) / point;

        if(profit >= triggerPips && currentSL < openPrice) {
            double newSL = NormalizeDouble(openPrice + lockPips * point, digits);
            ModifyStopLoss(ticket, newSL);
        }
    }
    else {
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double profit = (openPrice - ask) / point;

        if(profit >= triggerPips && (currentSL > openPrice || currentSL == 0)) {
            double newSL = NormalizeDouble(openPrice - lockPips * point, digits);
            ModifyStopLoss(ticket, newSL);
        }
    }
}
```

---

## Risk Budgeting

Allocate risk budget across strategies/timeframes:

```mql5
class CRiskBudget {
private:
    double totalDailyRisk;      // Total daily risk budget (%)
    double allocatedRisk;       // Currently allocated

    struct StrategyAllocation {
        string name;
        double maxRisk;         // Max risk for this strategy
        double usedRisk;        // Currently used
    };

    StrategyAllocation strategies[];

public:
    CRiskBudget(double dailyRisk = 3.0) {
        totalDailyRisk = dailyRisk;
        allocatedRisk = 0;
    }

    void AddStrategy(string name, double maxRisk) {
        int size = ArraySize(strategies);
        ArrayResize(strategies, size + 1);
        strategies[size].name = name;
        strategies[size].maxRisk = maxRisk;
        strategies[size].usedRisk = 0;
    }

    bool CanAllocate(string strategyName, double riskAmount) {
        // Check total budget
        if(allocatedRisk + riskAmount > totalDailyRisk)
            return false;

        // Check strategy budget
        for(int i = 0; i < ArraySize(strategies); i++) {
            if(strategies[i].name == strategyName) {
                if(strategies[i].usedRisk + riskAmount > strategies[i].maxRisk)
                    return false;
                break;
            }
        }

        return true;
    }

    void Allocate(string strategyName, double riskAmount) {
        allocatedRisk += riskAmount;

        for(int i = 0; i < ArraySize(strategies); i++) {
            if(strategies[i].name == strategyName) {
                strategies[i].usedRisk += riskAmount;
                break;
            }
        }
    }

    void Release(string strategyName, double riskAmount) {
        allocatedRisk -= riskAmount;

        for(int i = 0; i < ArraySize(strategies); i++) {
            if(strategies[i].name == strategyName) {
                strategies[i].usedRisk -= riskAmount;
                break;
            }
        }
    }

    void DailyReset() {
        allocatedRisk = 0;
        for(int i = 0; i < ArraySize(strategies); i++) {
            strategies[i].usedRisk = 0;
        }
    }

    double GetRemainingBudget() {
        return totalDailyRisk - allocatedRisk;
    }
};
```

---

## Emergency Procedures

### Panic Close All

```mql5
void EmergencyCloseAll(string reason = "") {
    Print("EMERGENCY CLOSE: ", reason);

    // Close all positions immediately
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        if(PositionSelectByIndex(i)) {
            ulong ticket = PositionGetInteger(POSITION_TICKET);
            ClosePositionMarket(ticket);
        }
    }

    // Cancel all pending orders
    for(int i = OrdersTotal() - 1; i >= 0; i--) {
        ulong ticket = OrderGetTicket(i);
        if(ticket > 0) {
            CancelPendingOrder(ticket);
        }
    }

    // Disable EA
    ExpertRemove();
}

void ClosePositionMarket(ulong ticket) {
    if(!PositionSelectByTicket(ticket)) return;

    MqlTradeRequest request = {};
    MqlTradeResult result = {};

    request.action = TRADE_ACTION_DEAL;
    request.position = ticket;
    request.symbol = PositionGetString(POSITION_SYMBOL);
    request.volume = PositionGetDouble(POSITION_VOLUME);
    request.deviation = 50;  // Wide deviation for emergency
    request.type_filling = ORDER_FILLING_IOC;

    if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
        request.type = ORDER_TYPE_SELL;
        request.price = SymbolInfoDouble(request.symbol, SYMBOL_BID);
    } else {
        request.type = ORDER_TYPE_BUY;
        request.price = SymbolInfoDouble(request.symbol, SYMBOL_ASK);
    }

    OrderSend(request, result);
}
```

### Connection Loss Handler

```mql5
datetime lastTickTime = 0;
int noTickWarningSeconds = 30;
int noTickPanicSeconds = 60;

void CheckConnection() {
    datetime now = TimeCurrent();

    if(lastTickTime > 0) {
        int elapsed = (int)(now - lastTickTime);

        if(elapsed > noTickPanicSeconds) {
            Print("WARNING: No tick for ", elapsed, " seconds. Connection may be lost.");
            // Consider reducing exposure or hedging
        }
        else if(elapsed > noTickWarningSeconds) {
            Print("Caution: No tick for ", elapsed, " seconds.");
        }
    }
}

void OnTick() {
    lastTickTime = TimeCurrent();
    // ... rest of OnTick
}
```
