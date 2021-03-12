//////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                              //
// Copyright (c) 2019 Tencent.com, Inc. All Rights Reserved                                     //
// Author: Aster Jian (asterjian@qq.com)                                                        //
//                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////
#include "simple_table.h"

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <sstream>

#ifdef _MSC_VER
#include <codecvt>
#endif

extern int mk_wcswidth_cjk(const wchar_t *pwcs, size_t n);

namespace utils
{
    /**
     * \brief �õ���ʾ�Ŀ���
     */
    inline int GetDisplayWidth(const std::string& str)
    {
#ifdef _MSC_VER
        std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
        std::wstring wstr = converter.from_bytes(str);
        return mk_wcswidth_cjk(wstr.c_str(), wstr.length());
#else
        wchar_t buff[1024]; // Ҫ����ʾ����С�� 1024���������ᱻ�ض�
        size_t len = mbstowcs(buff, str.c_str(), 1024);
        return mk_wcswidth_cjk(buff, len);
#endif
    }

    Column::Column(const std::string& name)
    {
        SetName(name);
    }

    Column& Column::SetName(const std::string& name)
    {
        name_ = name;        
        preferred_width_ = GetDisplayWidth(name);

        return *this;
    }

    Column& Column::SetPrecision(int precision)
    {
        precision_ = precision;
        return *this;
    }

    Column& Column::SetPercentage(bool percentage)
    {
        percentage_ = percentage;
        return *this;
    }

    /*Column& Column::SetWidth(int width)
    {
        setted_width_ = width;
        return *this;
    }

    Column& Column::SetAlignment(Alignment alignment)
    {
        setted_alignment_ = alignment;
        return *this;
    }
    */

    void Column::Print(std::ostream& os) const
    {
        Print(os, name_);
    }

    void Column::Print(std::ostream& os, const std::string& value) const
    {
        auto alignment = setted_alignment_ != Alignment::Default ? setted_alignment_ : preferred_alignment_;
        if (alignment == Alignment::Left)
        {
            os << std::left;
        }
        else
        {
            os << std::right;
        }

        int width = setted_width_ != 0 ? setted_width_ : preferred_width_;

        os << std::setw(width + value.length() - GetDisplayWidth(value)) << value;
    }

    Row::Row(std::vector<Column>& columns) :columns_(columns)
    {
    }

    Row& Row::AddBool(bool value)
    {
        Column& column = columns_[values_.size()];

        std::stringstream sstream;
        sstream << std::boolalpha << value;
        values_.push_back(sstream.str());

        column.preferred_width_ = std::max(column.preferred_width_, (int)sstream.str().length());

        return *this;
    }

    Row& Row::AddInt(int value)
    {
        Column& column = columns_[values_.size()]; 
        
        std::stringstream sstream;
        sstream << value;
        values_.push_back(sstream.str());

        column.preferred_width_ = std::max(column.preferred_width_, (int)sstream.str().length());

        return *this;
    }

    Row& Row::AddFloat(float value)
    {
        Column& column = columns_[values_.size()];

        std::stringstream sstream; 
        sstream << std::fixed << std::setprecision(column.precision_);
        if (column.percentage_)
        {
            sstream << value * 100 << "%";
        }
        else
        {
            sstream << value;
        }
        values_.push_back(sstream.str());
        
        column.preferred_width_ = std::max(column.preferred_width_, (int)sstream.str().length());

        return *this;
    }
    
    Row& Row::AddString(const std::string& value)
    {
        Column& column = columns_[values_.size()];
        
        values_.push_back(value);

        column.preferred_width_ = std::max(column.preferred_width_, GetDisplayWidth(value));
        column.preferred_alignment_ = Alignment::Left;

        return *this;
    }

    void Table::Print(std::ostream& os) const
    {
        if (rows_.empty())
        {
            return;
        }

        PrintHeader(os);

        for (const Row& row : rows_)
        {
            PrintRow(os, row);
        }

        PrintLine(os, '-');
    }

    void Table::PrintHeader(std::ostream& os) const
    {
        PrintLine(os, '-');

        os << "| ";
        for (int i = 0; i < columns_.size(); ++i)
        {
            columns_[i].Print(os);

            if (i != columns_.size() - 1)
            {
                os << " | ";
            }
        }
        os << " |" << std::endl;

        PrintLine(os, '-');
    }

    void Table::PrintLine(std::ostream& os, char ch) const
    {
        int total_width = std::accumulate(columns_.begin(),
            columns_.end(), (int)columns_.size() * 3 + 1, [](int w, const Column& column)
        {
            return w + (column.setted_width_ != 0 ? column.setted_width_ : column.preferred_width_);
        });

        for (int i = 0; i < total_width; ++i)
        {
            os << ch;
        }
        os << std::endl;
    }

    void Table::PrintRow(std::ostream& os, const Row& row) const
    {
        if (row.values_.size() != columns_.size())
        {
            return;
        }

        os << "| ";
        for (int i = 0; i < columns_.size(); ++i)
        {
            columns_[i].Print(os, row.values_[i]);
            if (i != columns_.size() - 1)
            {
                os << " | ";
            }
        }
        os << " |" << std::endl;
    }
}