########################################################################################
##
##                                  TESTS FOR 
##                              'blocks._block.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathlib import Path

from pathsim.blocks._block import Block
from pathsim.blocks.amplifier import Amplifier
from pathsim.utils.register import Register
from pathsim.utils.portreference import PortReference


# TESTS ================================================================================

class TestBlock(unittest.TestCase):
    """
    Test the implementation of the base 'Block' class
    """

    def test_init(self):

        B = Block()

        #test default inputs and outputs
        self.assertTrue(isinstance(B.inputs, Register))
        self.assertTrue(isinstance(B.outputs, Register))

        self.assertEqual(len(B.inputs), 1)
        self.assertEqual(len(B.outputs), 1)

        self.assertEqual(B.inputs[0], 0.0)
        self.assertEqual(B.outputs[0], 0.0)

        #test default engine not defined
        self.assertEqual(B.engine, None)

        #is active by default
        self.assertTrue(B._active)

        #operators not defined by default
        self.assertEqual(B.op_alg, None)
        self.assertEqual(B.op_dyn, None)


    def test_len(self):

        B = Block()

        #test default len method
        self.assertEqual(len(B), 1)
            

    def test_on_off_bool(self):
        
        B = Block()

        #default active
        self.assertTrue(B)

        #deactivate block
        B.off()
        self.assertFalse(B)

        #activate block
        B.on()
        self.assertTrue(B)


    def test_getitem(self):

        B = Block()

        #test default getitem method
        pr = B[0]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.block, B)
        self.assertEqual(pr.ports, [0])

        pr = B[2]
        self.assertEqual(pr.ports, [2])

        pr = B[30]
        self.assertEqual(pr.ports, [30])

        #test input validation
        with self.assertRaises(ValueError): B[0.2]
        with self.assertRaises(ValueError): B[1j]
        with self.assertRaises(ValueError): B["a"]


    def test_getitem_slice(self):

        B = Block()

        #test slicing in getitem
        pr = B[:1]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.ports, [0])

        pr = B[:2]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.ports, [0, 1])

        pr = B[1:2]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.ports, [1])

        pr = B[0:5]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.ports, [0, 1, 2, 3, 4])

        pr = B[3:7]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.ports, [3, 4, 5, 6])

        pr = B[3:7:2]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.ports, [3, 5])

        pr = B[:10:3]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.ports, [0, 3, 6, 9])

        pr = B[2:12:4]
        self.assertTrue(isinstance(pr, PortReference))
        self.assertEqual(pr.ports, [2, 6, 10])

        #slice input validation
        with self.assertRaises(ValueError): B[1:] #open ended
        with self.assertRaises(ValueError): B[:0] #starting at zero




    def test_reset(self):

        B = Block()

        B.inputs.update_from_array([1, 2, 3])
        B.outputs.update_from_array([-1, 5])


        #test if inputs and outputs are set correctly
        self.assertEqual(B.inputs[0], 1)
        self.assertEqual(B.inputs[1], 2)
        self.assertEqual(B.inputs[2], 3)

        self.assertEqual(B.outputs[0], -1)
        self.assertEqual(B.outputs[1], 5)


        B.reset()

        #test if inputs and outputs are reset correctly
        self.assertEqual(B.inputs[0], 0.0)
        self.assertEqual(B.inputs[1], 0.0)
        self.assertEqual(B.inputs[2], 0.0)

        self.assertEqual(B.outputs[0], 0.0)
        self.assertEqual(B.outputs[1], 0.0)


    def test_update(self):

        B = Block()

        #test default implementation 
        self.assertEqual(B.update(None), 0.0)


    def test_solve(self):

        B = Block()

        #test default implementation 
        self.assertEqual(B.solve(None, None), 0.0)


    def test_step(self):

        B = Block()

        #test default implementation (scale=None means no rescale needed)
        self.assertEqual(B.step(None, None), (True, 0.0, None))


    def test_info_base_block(self):
        """Test info method on base Block class"""

        info = Block.info()

        #check all expected keys are present
        expected_keys = {"type", "description", "input_port_labels", "output_port_labels", "parameters", "icon"}
        self.assertEqual(set(info.keys()), expected_keys)

        #check type is correct
        self.assertEqual(info["type"], "Block")

        #check description is the docstring
        self.assertIn("Base 'Block' object", info["description"])

        #check port labels (base Block has dynamic ports)
        self.assertIsNone(info["input_port_labels"])
        self.assertIsNone(info["output_port_labels"])

        #check parameters (base Block has no parameters)
        self.assertEqual(info["parameters"], {})

        #base Block has no icon
        self.assertIsNone(info["icon"])


    def test_info_with_parameters(self):
        """Test info method on block with parameters"""

        info = Amplifier.info()

        #check type is correct
        self.assertEqual(info["type"], "Amplifier")

        #check description contains relevant info
        self.assertIn("Amplifies", info["description"])

        #check port labels (Amplifier has dynamic ports)
        self.assertIsNone(info["input_port_labels"])
        self.assertIsNone(info["output_port_labels"])

        #check parameters include gain with default
        self.assertIn("gain", info["parameters"])
        self.assertEqual(info["parameters"]["gain"]["default"], 1.0)


    def test_info_caching(self):
        """Test that info method results are cached"""

        #clear cache first
        Block.info.cache_clear()

        #call info twice
        info1 = Block.info()
        info2 = Block.info()

        #should be same object due to caching
        self.assertIs(info1, info2)

        #check cache was used
        cache_info = Block.info.cache_info()
        self.assertEqual(cache_info.hits, 1)
        self.assertEqual(cache_info.misses, 1)


    def test_register_sizing_from_port_labels(self):
        """Registers are pre-sized to the declared port labels so blocks
        with unconnected multi-port interfaces stay positionally addressable.
        """

        class _LabeledBlock(Block):
            input_port_labels = {"a": 0, "b": 1}
            output_port_labels = {"x": 0, "y": 1, "z": 2}

        B = _LabeledBlock()

        #registers sized to highest declared index + 1
        self.assertEqual(len(B.inputs), 2)
        self.assertEqual(len(B.outputs), 3)

        #shape reflects declared ports even without connections
        self.assertEqual(B.shape, (2, 3))

        #all declared ports addressable and zero-initialized
        self.assertEqual(B.inputs["b"], 0.0)
        self.assertEqual(B.outputs["z"], 0.0)

        #blocks without port labels keep the default size 1 registers
        self.assertEqual(len(Block().inputs), 1)
        self.assertEqual(len(Block().outputs), 1)



class TestBlockIcon(unittest.TestCase):
    """
    Test the icons of blocks
    """

    def test_core_block(self):

        icon = Amplifier.icon()
        self.assertTrue(icon.startswith("<svg"))
        self.assertIs(Amplifier.info()["icon"], icon)


    def test_subclass_has_no_icon(self):

        class MyAmplifier(Amplifier):
            pass

        self.assertIsNone(MyAmplifier.icon())


    def test_package_icons(self):
        """a package ships icons for its blocks in its own 'icons' directory"""

        import sys
        import tempfile
        import importlib

        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "icontoolbox"
            (pkg / "icons").mkdir(parents=True)
            (pkg / "__init__.py").write_text(
                "from pathsim.blocks import Block\n"
                "class Widget(Block): pass\n"
                "class Gadget(Block): pass\n"
                )
            (pkg / "icons" / "Widget.svg").write_text('<svg viewBox="0 0 96 64"/>')

            sys.path.insert(0, tmp)
            try:
                mod = importlib.import_module("icontoolbox")
                self.assertEqual(mod.Widget.icon(), '<svg viewBox="0 0 96 64"/>')
                self.assertIsNone(mod.Gadget.icon())
            finally:
                sys.path.remove(tmp)
                sys.modules.pop("icontoolbox", None)


    def test_shipped_icons(self):
        """every shipped icon is valid SVG in the shared geometry and belongs to a block"""

        import xml.etree.ElementTree as ET
        import pathsim
        import pathsim.blocks

        blocks = {
            name: obj for name, obj in vars(pathsim.blocks).items()
            if isinstance(obj, type) and issubclass(obj, Block)
            }
        blocks.update(Subsystem=pathsim.Subsystem, Interface=pathsim.Interface)

        icons = sorted((Path(pathsim.__file__).parent / "icons").glob("*.svg"))
        self.assertGreater(len(icons), 0)

        for path in icons:
            with self.subTest(icon=path.stem):
                root = ET.fromstring(path.read_text(encoding="utf-8"))
                self.assertEqual(root.get("viewBox"), "0 0 96 64")
                self.assertFalse([e for e in root.iter() if e.tag.endswith("script")])
                self.assertIn(path.stem, blocks)
                self.assertIsNotNone(blocks[path.stem].icon())

# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)