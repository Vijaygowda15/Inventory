import pandas as pd

class InventoryManager:
    def __init__(self):
        # Initializing an empty DataFrame to hold the inventory data
        self.inventory = pd.DataFrame(columns=['Item_ID', 'Item Name', 'Category', 'Stock_Quantity', 'Price'])

    def add_item(self, item_id, item_name, category, stock_quantity, price):
        """Add a new item to the inventory."""
        new_item = {'Item_ID': item_id, 'Item Name': item_name, 'Category': category, 
                    'Stock_Quantity': stock_quantity, 'Price': price}
        self.inventory = self.inventory.append(new_item, ignore_index=True)

    def remove_item(self, item_id):
        """Remove an item from the inventory by its Item_ID."""
        self.inventory = self.inventory[self.inventory['Item_ID'] != item_id]

    def update_stock(self, item_id, new_quantity):
        """Update the stock quantity of a given item."""
        self.inventory.loc[self.inventory['Item_ID'] == item_id, 'Stock_Quantity'] = new_quantity

    def generate_low_stock_report(self, threshold):
        """Generate a report for items with stock below the threshold."""
        low_stock_items = self.inventory[self.inventory['Stock_Quantity'] < threshold]
        return low_stock_items

    def __repr__(self):
        return self.inventory.to_string(index=False)

# Example usage:
# inv_manager = InventoryManager()
# inv_manager.add_item(1, "Laptop", "Electronics", 10, 999.99)
# inv_manager.add_item(2, "Mouse", "Electronics", 5, 25.50)
# print(inv_manager)
# low_stock = inv_manager.generate_low_stock_report(6)
# print("Low Stock Report:")
# print(low_stock)
