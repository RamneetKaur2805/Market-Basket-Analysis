# Market Basket Analysis and Product Recommendation System

## Project Title

**Market Basket Analysis and Product Recommendation System using Apriori Algorithms**

---

# Business Problem

Retail businesses generate large volumes of transactional purchase data every day. However, identifying hidden relationships between products manually is difficult.

The company wants to understand:

- Which products are frequently purchased together
- Which products should be bundled
- Which products can improve cross-selling opportunities
- Which products should be placed near each other
- Which products can be recommended to customers

The objective of this project is to use **Market Basket Analysis** techniques to identify meaningful product associations and generate business recommendations that improve sales and customer experience.

---

# Dataset Description

The dataset used in this project is a retail transaction dataset containing customer purchase records.

The dataset includes columns such as:

| Column Name | Description |
|---|---|
| TransactionID | Unique identifier for each transaction |
| CustomerID | Unique customer identifier |
| TransactionDate | Date of each transaction |
| ProductID | Unique identifier for each Product |
| Product Name | Name of purchased product |
| Quantity | Quantity purchased |
| UnitPrice | Price per unit purchased |

---

# Data Understanding

## What is a Transaction?

A transaction represents a single customer purchase event identified using a transaction ID.

### Example

| Transaction ID | Purchased Products |
|---|---|
| T001 | Bread, Milk, Butter |

---

## What is an Item?

An item refers to an individual product purchased within a transaction.

### Examples
- Bread
- Milk
- Butter

---

## What Each Row Represents

Each row in the dataset represents one product purchased within a transaction.

If multiple products are purchased in one transaction, multiple rows will share the same transaction ID.

---

## Why Market Basket Analysis is useful for the business

It helps the businesses to identify products that customers frequently purchase together. By 

discovering these hidden purchasing patterns, companies can make better business decisions and 

improve overall sales performance.

---

## How product associations can help in cross-selling and upselling

### Cross – Selling:  

Cross-selling means recommending related or complementary products to customers based on their current purchase. 

Product associations help businesses identify which products are commonly bought together.

Example

If customers frequently buy:
	•	Dishwash Liquid → Fabric Softener

the business can recommend fabric softener whenever a customer purchases dishwash liquid. 

### Upselling: 

Upselling means encouraging customers to purchase a higher-value, premium, or upgraded version of a product.

Product associations help businesses identify opportunities where customers buying one product are also likely to purchase premium or additional products.

Example

If customers buying a basic coffee machine frequently purchase premium coffee capsules, the business can promote higher-end coffee products during checkout.

---

# Data Cleaning Summary

Before generating association rules, several cleaning operations were performed to improve data quality and analysis accuracy.

## Cleaning Steps Performed

### 1. Removed Missing Product Names

Rows with missing productname were removed.

```python
df = df.dropna(subset=['ProductName'])
```

---

### 2. Removed Duplicate Records

Duplicate transaction-product records were removed.

```python
df = df.drop_duplicates()
```

---

### 3. Removed Cancelled Transactions

Cancelled transactions identified using TransactionID beginning with “C” were removed.

```python
df = df[~df['TransactionID'].astype(str).str.startswith('C')]
```

---

### 4. Removed Invalid Quantities

Transactions with zero or negative quantities were removed.

```python
df[df['Quantity'] < 0 ]
```

---

### 5. Removed Rare Products

Products purchased very few times were removed to improve rule quality. But in the dataset there 

was no such item which was purchased for very few times, almost every item purchased greater than 

100 times.

```python
print([df['ProductName'].value_counts()])
```

---

### 6. Correcting Datatypes

Changing incorrect datatype to correct ones.

```python
df['TransactionID'] = df['TransactionID'].astype(str)
df['Quantity'] = df['Quantity'].astype(int)
df['UnitPrice'] = df['UnitPrice'].astype(float)
```

---

# Basket Preparation Method

The transactional dataset was converted into basket format suitable for Market Basket Analysis.

## Basket Transformation Process

The following steps were performed:

1. Grouped data using:
   - Transaction ID
   - Product Name

2. Converted transactional rows into matrix format

3. Applied binary encoding:
   - 1 = Product purchased
   - 0 = Product not purchased

---

## Basket Creation Code

```python
basket = (
    df.groupby(['TransactionID', 'ProductName'])['Quantity']
    .sum()
    .unstack()
    .fillna(0)
)

basket = basket.applymap(lambda x: 1 if x > 0 else 0)
```

---

## Basket Format Example

| Transaction ID | Bread | Milk | Butter |
|---|---|---|---|
| T001 | 1 | 0 | 1 |
| T002 | 0 | 1 | 1 |

---

# Frequent Itemset Summary

Frequent itemset were generated using the **Apriori Algorithm**.

Different support thresholds were plotted to observe changes in the number of itemset.

---

## Support Values Tested

| Support Value | Observation |
|---|---|
| 0.01 | Generated many itemset including weak combinations |
| 0.05 | Balanced and meaningful itemset |
| 0.1   | Generated fewer but stronger itemset |

A support value of **0.05** was selected because it provided meaningful product combinations without excessive noise.

---

## Choosing a reasonable support value and generating frequent itemsets

```python
min_support_chosen = 0.05
frequent_itemsets = apriori(basket, min_support=min_support_chosen, use_colnames=True)

print(f"\nFrequent Itemsets generated with min_support = {min_support_chosen}:")
display(frequent_itemsets.sort_values(by='support', ascending=False).head(10))
print(f"Total frequent itemsets: {len(frequent_itemsets)}")
)
```

---

# Association Rules Summary

Association rules were generated using:

- Support
- Confidence
- Lift

---

## Association Rule Generation Code

```python
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

print("Generated Association Rules (min_confidence = 0.6):")
display(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='confidence', ascending=False))
```

---

# Understanding Association Metrics

## Support: 

Support for an itemset measures how frequently the itemset appears in the dataset. A higher 

support value indicates that the itemset is more common. For example, if the support for {Milk, 

Bread} is 0.10, it means that 10% of all transactions contain both Milk and Bread.

### What support tells us: 

It indicates the popularity of an itemset. It helps filter out infrequent itemset, as rules involving very 

rare items might not be practically useful.

## Confidence: 

Confidence for a rule {A} -> {B} measures how often items in {B} appear in transactions that also 

contain items in {A}. It's the conditional probability of {B} given {A}. For example, if the confidence for 

{Milk} -> {Bread} is 0.80, it means that 80% of the transactions that contain Milk also contain Bread.

### What confidence tells us: 

It indicates the reliability of the rule. A high confidence suggests that the presence of the antecedent 

(A) strongly implies the presence of the consequent (B).

## Lift: 

Lift for a rule {A} -> {B} measures how much more likely items in {B} are to appear in transactions that 

contain items in {A}, compared to {B}'s general popularity (i.e., its probability of appearing in any 

transaction). It's the ratio of observed support to expected support if A and B were independent.

### What lift tells us: 

It indicates the strength of the association between A and B, beyond what would be expected by 

chance.

## Why lift greater than 1 is generally useful:

Lift = 1: Implies that the probability of occurrence of A and B is independent. There is no association 

between A and B.

Lift > 1: Suggests a positive correlation between A and B. The presence of A increases the likelihood 

of B appearing. The higher the lift, the stronger the positive association, making these rules 

potentially interesting for recommendations or placement strategies.

Lift < 1: Suggests a negative correlation between A and B. The presence of A decreases the likelihood 

of B appearing. This could indicate that A and B are substitute products.

---

# Top Rules with Interpretation

(Rules Interpretation is specified with the code in the notebook)

# Final Business Recommendations

Based on the interpreted association rules, here are practical recommendations to improve sales and customer experience:

---

## Product Bundling:

### 1) Dishwash Liquid & Fabric Softener: 

Given the strong association (Rule 1 & 2), these household cleaning items can be bundled together, perhaps with a slight discount, to encourage purchase of both.

### 2) Nachos & Salsa Dip: 

This is a classic pairing (Rule 3) and an ideal candidate for 'buy one get one half off' or a combined snack bundle. Special display units for these together could also be effective.

### 3) Potato Chips & Salsa Dip: 

Similar to nachos, these (Rule 4) can be bundled as a snack combo. Consider different flavors of chips with various dips.

### 4) Bread Loaf, Butter & Chocolate Spread: 

The rules involving these three items (Rules 5, 6, 7, 8, 9, 10) indicate a strong breakfast/snack bundle. Offering a 'breakfast starter kit' with all three could be very appealing.

## Product Placement:

### 1) Dishwash Liquid & Fabric Softener: 

Place these items in close proximity on shelves, even if they belong to different categories, to facilitate complementary purchases.

### 2) Nachos, Potato Chips & Salsa Dip: 

These snack items should always be displayed together. Consider end-cap displays or prominent placement in the snack aisle.

### 3) Butter, Bread Loaf & Chocolate Spread: 

These items should be merchandised near each other. If bread is in a separate section, consider placing a smaller display of bread near the butter and spreads.

## Cross-Selling Strategies:

### 1) Granola Bar & Oats Pack: 

(Not explicitly in the top 10 rules interpreted, but observed in previous frequent itemset analysis). Promote granola bars to customers purchasing oats, and vice-versa, as these are related healthy breakfast/snack items.

### 2) Coffee Beans & French Press: 

If a customer buys coffee beans, suggest a French press (or vice-versa). This can be done via online recommendations or in-store signage.

### 3) Instant Coffee & French Press: 

While less intuitive, the association (if present in the full rules) could indicate customers exploring different coffee brewing methods. A recommendation for one when the other is purchased could introduce them to new options.

## Promotional Campaigns:

### 1) The 'Breakfast Trio' (Bread Loaf, Butter, Chocolate Spread): 

Run targeted promotions for these items during morning hours or on weekends.

### 2) 'Movie Night' or 'Game Day' Snack Packs (Nachos/Potato Chips + Salsa Dip): 

Create themed promotions around events where these items are commonly consumed.

### 3) 'Household Essentials' Bundle (Dishwash Liquid + Fabric Softener): 

Offer seasonal discounts on these essential cleaning products.

## Rules to Ignore (Despite High Metrics):

### 1) Inverse Rules:
 
 It's important to recognize that rules like A -> B and B -> A (e.g., Dishwash Liquid -> Fabric Softener and Fabric Softener -> Dishwash Liquid) provide similar insights. While both are valid, the interpretation focuses on the primary direction that might be more actionable for marketing (e.g., if one item is a 'destination' purchase).

### 2) Obvious or Trivial Rules: 

Rules that are extremely common knowledge (e.g., Bread -> Butter if the confidence is very high) might not offer new strategic insights but confirm existing assumptions. However, they can still be valuable for consistent merchandising.

## Improving Sales or Customer Experience:

### 1) Increased Basket Size: 

By bundling and strategically placing complementary products, customers are encouraged to purchase more items in a single transaction, directly increasing sales.

### 2) Enhanced Shopping Convenience: 

Customers save time by finding related products together, leading to a more positive shopping experience.

### 3) Personalized Recommendations: 

Online stores can use these rules to power 'Customers who bought this also bought...' sections, leading to more relevant suggestions and increased conversions.

### 4) Optimized Inventory Management: 

Understanding co-purchase patterns can help in stocking decisions, ensuring that complementary items are always available together.

### 5) Targeted Marketing: 

The insights enable more effective marketing campaigns, focusing on product relationships rather than individual items, which can resonate more with customer buying habits

---

# Visualizations

The project includes visualizations such as:

- Number of frequent itemset vs minimum Support
- Association Rules: Support vs Confidence 

---

# Project Structure

```bash
Market-Basket-Analysis/
│
├── Market_Basket_Analysis.ipynb
├── README.md
├── requirements.txt
├── dataset_source.md
├── outputs/
└── images/```

---

# How to Run the Project

## Step 1: Clone Repository

```bash
git clone <repository-link>
```

---

## Step 2: Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## Step 3: Open Jupyter Notebook or Google Colab

Open:

```bash
Market_Basket_Analysis.ipynb
```

---

## Step 4: Run All Cells

Execute all notebook cells sequentially.

---
