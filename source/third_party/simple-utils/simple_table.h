//////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                              //
// Copyright (c) 2019 Tencent.com, Inc. All Rights Reserved                                     //
// Author: Aster Jian (asterjian@qq.com)                                                        //
//                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace utils
{
    /**
     * \brief 对齐方式
     */
    enum class Alignment
    {
        Default,
        Left,
        Right,
    };

    /**
     * \brief 列定义
     */
    struct Column
    {
        /**
         * \brief
         */
        Column(const std::string& name);

        /**
         * \brief 设置名称
         */
        Column& SetName(const std::string& name);

        /**
         * \brief 设置精度（小数点位数）
         */
        Column& SetPrecision(int precision);

        /**
         * \brief 设置百分数
         */
        Column& SetPercentage(bool percentage);

        /**
         * \brief 手动设置宽度（请不要调用，自动设置的会比较好）
         */
         // Column& SetWidth(int width);

        /**
         * \brief 手动设置对齐方式（请不要调用，自动设置的会比较好）
         */
        // Column& SetAlignment(Alignment alignment);

        /**
         * \brief 打印函数
         */
        void Print(std::ostream& os) const;

        /**
         * \brief 打印字符串
         */
        void Print(std::ostream& os, const std::string& value) const;

        /**
         * \brief
         */
        std::string name_;

        /**
         * \brief 精度（小数点位数）
         */
        int precision_{ 3 };

        /**
         * \brief 百分数
         */
        bool percentage_{ false };

        /**
         * \brief 手动设置的宽度
         */
        int setted_width_{ 0 };

        /**
         * \brief 自动设置的宽度
         */
        int preferred_width_{ 0 };

        /**
         * \brief 手动设置的对齐方式
         */
        Alignment setted_alignment_{ Alignment::Default };

        /**
         * \brief 自动设置的对齐方式
         */
        Alignment preferred_alignment_{ Alignment::Right };
    };

    /**
     * \brief 行定义
     */
    struct Row
    {
        /**
         * \brief 构造函数
         */
        Row(std::vector<Column>& columns);

        /**
         * \brief 添加布尔数值
         */
        Row& AddBool(bool value);

        /**
         * \brief 添加整形数值
         */

        Row& AddInt(int value);

        /**
         * \brief 添加浮点数值
         */
        Row& AddFloat(float value);

        /**
         * \brief 添加字符串数值
         */
        Row& AddString(const std::string& value);

        /**
         * \brief 所有的列引用
         */
        std::vector<Column>& columns_;

        /**
         * \brief 变成字符串后的数值
         */
        std::vector<std::string> values_;
    };

    /**
     * \brief 表格类
     */
    class Table
    {
    public:
        /**
         * \brief
         */
        Table() = default;

        /**
         * \brief
         */
        ~Table() = default;

        /**
         * \brief 添加列
         */
        Column& AddColumn(const std::string& name)
        {
            columns_.push_back(Column(name));
            return columns_.back();
        }

        /**
         * \brief 添加行
         */
        Row& AddRow()
        {
            rows_.push_back(Row(columns_));
            return rows_.back();
        }

        /**
         * \brief 打印函数
         */
        void Print(std::ostream& os = std::cout) const;

    private:
        /**
         * \brief 绘制表头
         */
        void PrintHeader(std::ostream& os) const;

        /**
         * \brief 绘制横线
         */
        void PrintLine(std::ostream& os, char ch) const;

        /**
         * \brief 绘制一行
         */
        void PrintRow(std::ostream& os, const Row& row) const;

        /**
         * \brief 表头列信息
         */
        std::vector<Column> columns_;

        /**
         * \brief 表格每行数据
         */
        std::vector<Row> rows_;
    };
}
 