from unittest import TestCase
import utils.check as check


class TestCheck_sep(TestCase):
    def test_check_sep_empty(self):
        name = ''
        actual = check.check_sep(name)
        excepted = ''

        self.assertEqual(actual, excepted)

    def test_check_sep_None(self):
        name = None
        actual = check.check_sep(name)
        excepted = ''

        self.assertEqual(actual, excepted)

    def test_check_sep_sep(self):
        name = 'abc'
        actual = check.check_sep(name, '/')
        excepted = '/abc'

        self.assertEqual(actual, excepted)
