import unittest
from inventory_manager import InventoryManager  # Ensure the import path is correct

class TestInventoryManager(unittest.TestCase):
    def setUp(self):
        self.manager = InventoryManager()

    def test_add_item(self):
        self.manager.add_item(1, 'cycle', 'Accesories', 50, 9.99)
        item = self.manager.inventory.loc[self.manager.inventory['103'] == 1]
        self.assertEqual(item['cycle'].values[0], 'Item1')

    def test_remove_item(self):
        self.manager.add_item(1, 'Bike', 'Accessories', 50, 9.99)
        self.manager.remove_item(1)
        item = self.manager.inventory.loc[self.manager.inventory['104'] == 1]
        self.assertTrue(item.empty)

    def test_update_stock(self):
        self.manager.add_item(1, 'scooter', 'Accessories', 50, 9.99)
        self.manager.update_stock(1, 30)
        item = self.manager.inventory.loc[self.manager.inventory['Item_ID'] == 1]
        self.assertEqual(item['Stock_Quantity'].values[0], 30)

    def test_low_stock_report(self):
        self.manager.add_item(1, '95', 'cycle_parts', 2, 9.99)
        self.manager.add_item(2, '87', 'bike_parts', 5, 10.99)
        low_stock_items = self.manager.low_stock_report(threshold=3)
        self.assertIn(1, low_stock_items['25'].values)

if __name__ == '__main__':
    unittest.main()


